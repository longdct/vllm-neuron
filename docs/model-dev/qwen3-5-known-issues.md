# Qwen3.5 / Qwen3.8 on Neuron: known issues

A scannable index of what does not work yet, or cannot on this hardware, for
the Qwen3.5-family text decoder (`vllm_neuron/model/qwen3_5/`). The narrative
docs (`qwen3-5-real-checkpoint-bringup.md`, `qwen3-5-tp8-device-bringup.md`)
record how each of these was found; this page exists so a reader does not
have to reconstruct that from the investigation logs. Each entry says what is
true today, why, and what -- if anything -- would change it.

## Hardware-gated

### FP8 weights do not run on Trainium2

Implementation is complete, unit-tested, and verified correct on CPU against
the real 0.8B checkpoint (84 quantized parameters, 2.65% worst reconstruction
error). It cannot run on Trn2: the prefill (CTE) MLP kernel sets its
PE-transpose destination dtype to match the quantized weight as soon as
weights are FP8, while the hidden state being transposed is still BF16 --
`nc_matmul (transpose mode) dst dtype must match input dtype on gen3+`. This
is true for both `QuantizationType.ROW` (no calibration needed) and `STATIC`
(needs a calibrated `input_scale`) -- measured, not assumed. `factory.py`
refuses `quantization="fp8"` at startup on trn2, in seconds, with this
explanation, rather than after a multi-minute compile and a NKI assertion.

**Path forward, not started:** `STATIC` FP8 + calibrated `input_scale`
tensors (e.g. via LLM-Compressor) + a fused RMSNorm-quant on the prefill path,
mirroring llama3's `model_static_fp8.py`. Needs calibration data no released
Qwen3.5/3.8 checkpoint ships. Unaffected on Trn3.

See "FP8 weights" in `qwen3-5-real-checkpoint-bringup.md` for the full
platform matrix and the bit-level Trn2-vs-OCP e4m3 comparison.

### TP=16 is unreachable for this family

Not a bug to fix -- a hard ceiling. vLLM requires `num_attention_heads % tp
== 0`; the 27B has 24 attention heads (`24 = 2^3 * 3`), so 8 is the largest
usable power of two regardless of hidden size or intermediate size. TP=8 is
therefore the maximum degree, not a starting point. A latent Q-head-padding
path (`padded_q_heads`, `q_head_indices`, `kv_head_index`) exists and is
correct, but nothing in the vLLM engine can reach it -- the head-divisibility
check runs first.

## Coverage gaps

### GatedDeltaNet layers are never quantized

48 of the 64 layers (20.7% of served weights) use plain `nn.Linear` for their
five projections, with no kernel-level quantization path. Under
`quantization="fp8"` they stay BF16 regardless -- only the MLP and
full-attention projections (69.9% of weights) are covered. A GDN quant path
would need a new dequant/quant kernel wrapper; not started.

### FP4 is not implemented

Not a gap so much as a non-target: no official FP4 checkpoint exists for
Qwen3.5/3.8 (the quantized release, `Qwen/Qwen3.8-27B-FP8`, is FP8), and
Trn2 has no FP4 tensor-engine format to target even if one existed. An FP4
`quantization_config` is refused by name at startup rather than silently
misread -- see "FP4 is not applicable and not implemented" in the bringup doc.

### Multi-token prediction (MTP) is not implemented

Both released checkpoints (0.8B and 27B) ship 15 `mtp.*` tensors and declare
`mtp_num_hidden_layers=1`. Nothing builds an MTP subtree; those weights are
skipped by prefix at load time, same as `model.visual.*`. The model serves
correctly as a plain decoder -- speculative decoding is simply unavailable.
`factory.py::_validate_config` warns about this rather than raising, since
raising would make every real checkpoint unservable.

### Vision input is not served

The text decoder only; no vision tower is built. Every released checkpoint
carries a `vision_config`, so the runner always supplies a
`vision_neuron_config` -- accepted so the checkpoint loads, then discarded
with a warning. `model.visual.*` weights are skipped by prefix. Image and
video inputs do not work.

### Chunked/segmented prefill is unavailable

`head_dim=256` exceeds the segmented-attention kernel's 128-element partition
bound, and `Qwen3_5Attention.forward_prefill` has no segmented branch at all
(unlike `qwen3`, which dispatches to `NF.segmented_attention`). Handed a
chunk, it would attend within that chunk only and silently ignore everything
cached before it -- coherent, confident, wrong output. `factory.py` raises at
startup if `kv_segment_size_buckets` is set (i.e. whenever
`max_num_batched_tokens < max_model_len`); single-shot prefill
(`max_num_batched_tokens == max_model_len`) is required.

## Performance, not correctness

### GDN prefill runs the torch chunk rule

The scan kernel is exact on device in isolation at every geometry, dtype,
decay regime, padding, and TP degree tried, yet costs 51 tokens at TP=1
(128 -> 77) once the 18 GDN call sites in the full graph interact with the
interleaved attention layers. Prefill falls back to the (slower) torch chunk
rule rather than the device kernel until that scheduling interaction is
understood. See "Still open" in the bringup doc.

## Known bug, not yet fixed

### Vision auto-config synthesizes a bucket floor even for text-only runs

`_resolve_vision_auto_config` (`vllm_neuron/vllm/platform.py`) synthesizes a
`vision_attention_block_size` of 2048 from the mere presence of
`vision_config` in the checkpoint, then rejects any smaller token bucket:
`ValueError: Largest bucket (256) is smaller than vision_attention_block_size
(2048)`. Workaround: raise `max_model_len` to >= 2048. The durable fix is
letting a text-only model opt out -- shared code with every multimodal model
on this backend, so it touches more than `qwen3_5`.

## Explicitly out of scope for this work

**On-device sampling** was part of the original two-goal investigation for
this effort (support sampling mode on Neuron; support FP8 weights) but was
dropped from scope -- another workstream owns it. Noted here only so its
absence from this list is not mistaken for "solved" or "forgotten": the
plugin's on-device sampling infrastructure exists and is on by default
(`on_device_sampling_config`); Qwen3.5's own test harness explicitly disables
it only because building a `Sampler` needs a live TP process group that a
unit test does not have.
