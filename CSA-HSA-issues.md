# DeepSeek-V4 HCA/CSA compiler issue: corrected attribution

Status: boundary-only NKI compressor implemented and structurally verified;
full-model timing and TP2/EP2 acceptance remain open.

## Correction

The earlier HCA plan attributed the 524,288-scale compiler fan-out to the MLA
compressed-history gather and predicted that a capacity-sized uniform HCA span
would remove it. Controlled whole-graph captures disproved that claim: the HCA
entry count and `module.mlir` changed, but every `Unrolled DGE count with
Dynamic AP` stayed identical.

DMA/source attribution identifies the large indirect load as
`attention.py::gather_recent_window_batched`, called from
`DeepseekV4Compressor.forward_packed`. The packed path projected and scattered
all raw states, gathered a complete raw window for every query, reduced every
query, and discarded non-boundaries with `-1` output slots. At Q512 this creates
the large HCA gather associated with `sg06`; CSA repeats it for both the
512-dimensional outer compressor and its independent 128-dimensional indexer
compressor, associated with `sg04`.

The HCA capacity rule and uniform MLA compressed span remain useful secondary
optimizations. They must be described and measured separately; they are not the
fix for the packed-compressor dependency explosion.

## Implemented boundary-only design

`vllm_neuron/model/deepseek_v4/nki_compressor.py` performs only paged FP32
raw-state gathering and gated reduction. Torch still owns projection/state
insertion, RMSNorm, RoPE, and compressed-cache insertion.

For each packed call the wrapper derives at most one candidate for Q1 and
`ceil(Q / ratio)` candidates otherwise. It advances candidates by the
compression ratio and validates index range, contiguous absolute position,
request ownership, compressed output mapping, and raw paged-state mapping.
Only `[candidate_count, coff * ratio]` slots enter NKI:

* HCA c128 reduces one non-overlapping 128-row window at head dimension 512.
* CSA c4 reduces the prior window's Ca half plus the current window's Cb half
  at head dimensions 512 and 128.

The kernel uses one `nl.fori_loop` runtime body with register-offset access,
LNC1 for a single candidate, and LNC2 otherwise. Missing early history is
masked before softmax; invalid candidates use finite sanitized inputs and emit
zero. `VLLM_NEURON_DSV4_NKI_COMPRESSOR=0` selects the portable fallback.

## Acceptance checkpoints

The simulator and portable tests establish numerical semantics, page crossing,
shuffled/null blocks, early overlap, and padded-tail handling. Compiler impact
must be established independently:

1. HCA: remove the `attention.py` 524,288-instance load and reduce `sg06` DGE
   count by at least 90%.
2. CSA outer/indexer: remove the 524,288- and 131,072-instance loads and reduce
   `sg04` DGE count by at least 90%.
3. Remove the dependency node reporting `Max Readers: 262,145`.
4. Confirm compressor COLZ/allocation structure does not scale linearly with Q
   and contains no `[Q, window, state_width]` allocation.
5. Only after component and one-rank checks, run one isolated TP2/EP2 cold
   compile. Acceptance remains under 600 seconds and 22 GiB compiler RSS,
   official tokens `[2030, 32974, 63376, 76010]`, selected-logit absolute error
   at most 0.125, and no warm-launch recompilation.

## Measurements on 2026-08-28

Cold production-FP32 component compiles succeeded for HCA Q1/Q512/Q1024,
CSA-512 Q512, and CSA-indexer-128 Q512. HCA Q512 and Q1024 produced 7,945- and
7,954-byte COLZ files with the same 130 allocation records; the artifact does
not scale linearly with Q and has no query-by-window state allocation.

A bounded dummy-weight TP1 Q512 whole-graph capture reduced the observed
Dynamic-AP DGE counts to about 2,000-2,300, more than 99% below the previous
262,408/327,808 counts. The `Max Readers: 262,145` node was absent and compiler
peak RSS was 15.4 GiB. The remaining outer graph did not finish within the
20-minute diagnostic bound, however, so checkpoint 5 is not satisfied. No
TP2/EP2 compile or official-token/warm-cache acceptance run was started.
