# DeepSeek-V4: blockers between a correct scaled-TP model and deployment

> **Status (2026-08-31):** Q8192 prefill and decode are validated at TP=8.
> The grouped output projection now supports TP=16/32/64, TP=16 has run at
> Q8192, and TP=32/EP=2 has run on device. TP=64 is structurally covered but
> was not run because only 32 logical cores were available without disturbing
> another workload. Multi-request decode is now validated at TP=8 with a live
> ragged batch of eight requests. Remaining deployment work includes production
> depth, compile cost, and the Neuron scheduler's existing single-prefill / no
> mixed-prefill-and-decode limitation.

## 2026-08-31 scale-first implementation update

This pass deliberately stays on the model's 16-bit BF16 path. FP8/FP4,
compile-bucket policy, and speculative-serving work are outside its scope.

The first gate was an unchanged-code TP=8 run with an 8192-token prompt. A cold
compile and a warm-cache repeat both completed a Q8192 prefill, Q8192 and Q32768
decode graphs, a first-token probe, and 32 generated tokens. The warm run
reproduced all 32 cold-run token IDs exactly.

| gate | result |
|---|---|
| TP=8, 3 layers / 32 experts, Q8192, cold | pass; 629.72 s compiler time; 3.17 tokens/s |
| TP=8, same cache and workload, warm | pass; identical 32 tokens; 3.97 tokens/s |
| TP=8, post-change Q8192 recompile | pass; exact same probe and 32 tokens; 3.98 tokens/s |
| TP=16, 3 layers / 32 experts, Q8192, cold + warm | pass; deterministic across cold/warm; 3.97 tokens/s warm |
| TP=32/EP=2, 3 layers / 32 experts, Q512 | pass; tokens exactly match the TP=16 Q512 control |
| TP=32/EP=2, 3 layers / **256 experts**, Q512 | pass; 1.315 GiB parameters and 1.35 GiB HBM used per rank; 2.60 tokens/s |

The TP>8 implementation splits each output group's `o_a` input columns among
consecutive ranks and replicates the matching `o_b` columns. The existing TP
all-reduce reconstructs the within-group partial sums. Unit tests cover the
partition map, checkpoint row/column slices, and numerical reconstruction at
TP=1/8/16/32/64. TP=32 uses EP=2 so the expert-TP degree remains 16; TP=64 is
validated with EP=4 by topology, loader, and numerical tests.

The TP=16 Q8192 sequence shares the first-token probe and first 10 generated
tokens with TP=8, then differs at BF16 router-sensitive positions. It is exactly
repeatable on a warm run. The stronger EP-specific check uses identical Q512
geometry: TP=32/EP=2 and TP=16 produced the same eight token IDs.

Expert-count scaling was checked with an official-derived 3-layer checkpoint
containing all 256 experts. The checkpoint verifier found all expected experts
in each layer and exact sampled dequantization against the official converter.
The TP=32/EP=2 cold run compiled Q512 prefill/decode in 248.65 s and completed
generation. This isolates the dominant checkpoint-width scaling axis; depth is
still limited to 3 of 43 layers by the locally available official shards.

## Where the model actually is

Two defects were found and fixed:

| commit | defect |
|---|---|
| `1194811` | Batch-1 decode hung at 256/8/8192. The indexer's page loop used a runtime register bound (`nl.fori_loop(0, page_count_reg, ...)`), forcing data-level LNC lockstep. Replaced with a static-control-flow decode kernel for Q<=8. |
| `254cee0` | Routed-expert weights were mis-sharded at every `tp_degree > 1`. `resolve_parallel_topology()` never plumbed `expert_tp_rank` on the non-EP path, so all ranks loaded shard 0 and the TP all-reduce summed `tp_degree` copies of one partial. |

The MoE fault is worth remembering as a *class* of bug: it was invisible to the
existing loader test, which passed `expert_tp_rank` in by hand. Only a test that
drives `resolve_parallel_topology` end to end catches it. That is now
`test_parallelism.py::test_routed_expert_shards_tile_the_checkpoint_under_resolved_topology`.

**Device verification after both fixes** (3-layer official-derived checkpoint,
greedy, `temperature=0.0`, 256-token prompt, only `world_size` differing):

| run | result |
|---|---|
| TP=8, ctx=2048, 8 tokens | 8/8 tokens identical to TP=1 |
| TP=8, ctx=2048, 32 tokens | 32/32 identical; 736 tensors compared, 0 shape mismatches |
| TP=8, ctx=32768, 32 tokens | 32/32 identical; segmented prefill; 0 NRT/NCC errors |

Residual per-layer error sits at a median of **1 BF16 ULP**, matching the noise
profile of attention (whose sharding is independently correct). About **0.06% of
rows** differ by >5%; these are top-k **router ties** — a BF16 near-tie flipping
which expert a token selects. Layer 0, which sees identical inputs, has 0 such
rows out of 2048. This is inherent to a BF16 top-k under any change in reduction
order and is not fixable by sharding.

> **Consequence for testing:** judge TP comparisons on **median-row relative RMS**
> and token equality. Worst-element error is dominated by these ties and is a
> useless gate.

## The target model

`/home/ubuntu/dsv4-official-shards/config.json` — DeepSeek-V4-Flash:

| | |
|---|---|
| num_hidden_layers | **43** (tiny model uses 3) |
| n_routed_experts | **256** (tiny uses 32) |
| num_hash_layers | 3 |
| hidden_size / head_dim | 4096 / 512 |
| num_attention_heads | 64 |
| o_groups | 8 |
| moe_intermediate_size | 2048 |
| vocab_size | 129280 |
| approx BF16 size | **~118 GB** |

118 GB across 24 GiB logical cores is ~14.8 GB/rank at TP=8 — tight once KV and
activations are added — but ~7.4 GB/rank at TP=16. **TP=16 is a practical
requirement, not an optimisation.**

---

## The four blockers

### 1. Multi-request model execution — resolved for decode

The original implementation had two live guards that raised on any batch:

```
model.py:605   "DeepSeek-V4 NKI CSA milestone supports one TP1 request"
model.py:1409  "DeepSeek-V4 NKI HCA milestone supports one TP1 request"
```

Both triggered on `block_tables.shape[0] != 1` or a non-zero
`token_to_request`. They are now removed. CSA and HCA retain the full 2-D block
table, and the indexer receives one owner per query so each query resolves its
own physical pages.

The portable batch-aware paths were used as the CPU oracle:

- `model.py:617-623` — `read_compressed_history_batched(mla_cache, block_tables, token_to_request, ...)`
- `model.py:1417-1424` — `logical_to_physical_slots_batched(..., owners, ...)`
- `nki_compressor.py:308` — already validates `token_to_request.shape == positions.shape`

The NKI indexer now accepts `[num_requests, num_blocks]` tables and a
`token_to_request` owner vector for both the static decode and runtime prefill
kernels. The compressor builds fixed candidate segments per request, preventing
one request from selecting another request's candidates. Cross-request MLA span
reuse and uniform compressed-history shortcuts are disabled when multiple block
table rows are present. Decode buckets 2 and 4 are accepted in addition to the
existing 1/8-and-larger buckets, and padded owner rows are masked inert.

The Neuron scheduler is a separate boundary. It still explicitly permits only
one prefill request at a time and does not mix prefill and decode in the same
step. The kernel interfaces are owner-aware for prefill, but this change does
not claim mixed-prefill scheduling support.

**Device result:** a TP=8 Q8 graph served a ragged batch of eight requests with
prompt lengths `[4,3,3,2,2,1,1,1]`, producing four tokens each. With the KV GMU
budget cap set to 1.0, the batched generation took 0.932 s (34.33 aggregate
tokens/s), and every output exactly matched an individual sequential run. The
same graph was also exercised under the default cap, where only two requests
were live and six graph rows were padding. A standalone real-Neuron B2/Q16
indexer trace additionally passed page-range and candidate-count checks.

### 2. Depth validated at 3 of 43 layers

Nothing — MoE sharding, static indexer, segmented prefill — has run at
production depth.

**Checkpoint availability is the constraint.** Only **5 of 46 shards** are on
disk, covering exactly layers 0, 2, 3 (the current tiny model's default slice).

| ladder depth | shards missing | download |
|---|---|---|
| 0..7 | 6 | ~15 GB |
| **0..15** | **14** | **~36 GB** |
| 0..42 (full) | 41 | ~105 GB |

152 GB free. A mid-depth ladder was chosen.

### 3. Compile cost

700s for **3 layers** at 32K, TP=8. Earlier context sweeps were distinctly
non-linear (TP=1: 1024 → 28.5 min, 2048 → 49.6 min, 4096 → >4 h), so linear
extrapolation to 43 layers is not safe in either direction.

**Compile is a cartesian product** — `neuron_worker.py:1680-1707`:

```
num_seqs_buckets x (decode_ctx_buckets + [max_model_len])
```

Today `num_seqs_buckets = [1]`. `get_default_num_seqs_buckets(8)` returns
`[1,2,4,8]`, so **enabling batching multiplies decode compiles by 4x**. Issues 1
and 3 are coupled: cut compile cost first, or pay 4x for every later experiment.

Compile is host-CPU-bound and already saturated (~28 concurrent `neuronx-cc`,
`walrus_driver` at ~700% CPU on 192 vCPUs), so more host parallelism is not
available as a lever.

### 5. TP ceiling — resolved in code through TP=64

There were two independent ceilings:

**`o_groups`** — `parallel.py` `validate()` rejects `output_groups % tp_degree`.
With `o_groups = 8`, TP=16 fails outright.

**A second, undocumented ceiling.** `model.py:1777-1782` requires per-rank expert
intermediate `>= 128`. With `moe_intermediate_size = 2048` that means
**`expert_tp_degree <= 16`**. Since `expert_tp_degree = world_size // ep_degree`,
TP=32 requires `ep_degree >= 2` and TP=64 requires `ep_degree >= 4`.
**Expert parallelism is a prerequisite for TP>16, not an independent feature.**

The output projection ceiling is now removed by sharding within each output
group when TP exceeds `o_groups`. TP=16 and TP=32/EP=2 have passed device
generation; TP=64/EP=4 passes topology, loader-slicing, and numerical tests.
TP=64 still needs a device run when all 64 logical cores are available.

---

## Plan

### Step 0 — Fetch shards (background, start first)

```bash
python tools/deepseek_v4/fetch_official_shards.py /home/ubuntu/dsv4-official-shards \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

~36 GB. `fetch_official_shards.py` already resolves which shards a layer list
needs. Independent of all code work.

### Step 1 — Lift the TP ceiling (implemented 2026-08-31)

**Files:** `model/deepseek_v4/parallel.py`, `model.py`,
`test_parallelism.py`

1. **Relax the `o_groups` guard** to permit both regimes: **done**
   - `world_size <= o_groups`: whole groups per rank (today), `o_groups % world_size == 0`
   - `world_size > o_groups`: one group split across `world_size // o_groups`
     ranks, requiring `world_size % o_groups == 0` and per-rank input width above
     the NKI partition floor.

2. **Generalise the output projection** (`model.py:827-843`, forward `:1265-1271`): **done**.
   `DeepseekV4GroupedLinear` (`model.py:717-732`) applies `F.linear` per group with
   **no bias**, and the forward already ends in `tp_group.all_reduce(out)`.
   Splitting a group's *input* across ranks is therefore exactly summable by that
   existing all-reduce — the same argument that makes head-sharding correct. Add
   `groups_per_rank` plus an input offset.

3. **Extend the `o_a_proj` weight loader** to slice dim 1 (within-group input) by
   the rank's position inside its group, alongside existing dim-0 group slicing.
   **Done**, with TP=16/32/64 checkpoint-slice tests.

4. **Wire EP for TP>16.** The existing machinery (`parallel.py` EP branch,
   `get_neuron_ep_tp_group`, `wide_ep_group`) now runs at `ep_degree=2` on
   device. Validate `ep_degree=4` at TP=64 when the full host is available; confirm
   `num_experts=256 % ep_degree == 0` and per-rank expert intermediate stays >=128.
   TP=32/EP=2 is now device-validated; TP=64/EP=4 remains a device-only follow-up.

Every new sharding path needs a test driving `resolve_parallel_topology` end to
end, asserting per-rank shards **tile the checkpoint exactly once**. See the note
on the `254cee0` bug class above.

### Step 2 — Cut compile cost

Before batching, because batching multiplies it by 4x.

1. **Make the implicit `max_model_len` decode bucket opt-out**
   (`neuron_worker.py:1704-1707`). When `decode_context_length_buckets` is given
   explicitly, appending `max_model_len` silently adds a compile per batch bucket.
2. **Measure before optimising.** `tools/deepseek_v4/analyze_compile_artifacts.py`
   and `count_graph_callsites.py` already exist. Establish instruction counts and
   wall-clock at 3, 8, 16 layers to get the real scaling exponent.
3. **Investigate per-layer graph reuse.** 43 layers span only 3 attention types.
   If structurally identical layer subgraphs are being re-derived, deduplication
   is the largest available win. Confirm feasibility before committing.
4. **Check compiler flags** (`NEURON_CC_FLAGS`, optlevel) for a compile/runtime
   trade-off acceptable during bring-up.

### Step 3 — Continuous decode batching (implemented 2026-08-31)

**Files:** `model.py`, `nki_indexer.py`, `nki_mla.py`

1. **Give the indexer a batch dimension: done.** Replaced the 1-D block-table
   contract with a `[num_seqs, num_blocks]` table plus a per-query owner vector
   in both decode and runtime-prefill kernels.

   The scheduler inspection confirmed that Neuron currently restricts prefill
   batches to one request and forbids mixed prefill/decode. Prefill nevertheless
   receives owners so the kernel boundary does not encode that scheduler limit.

2. **Lift the two guards: done.** CSA and HCA pass complete block tables and
   owner vectors instead of row zero.

3. **Test against the portable oracle: done.** The batched paths match
   `read_compressed_history_batched` / `logical_to_physical_slots_batched` for
   ragged batches — differing lengths, a zero-length request, a just-admitted
   request — in CPU unit tests and NKI simulator tests.

4. **Bucket integration: done.** The tool accepts explicit sequence buckets,
   decoder buckets 2 and 4 are legal, and tests confirm invalid padded owners
   stay inert. The TP=8 device run exercised both eight live rows and the
   default-budget case with six padded rows.

### Step 4 — Depth ladder

Rungs **3 → 8 → 16 layers** at TP=16 with batching enabled:

```bash
python tools/deepseek_v4/build_tiny_from_official.py \
    /home/ubuntu/dsv4-official-shards <out> --layers 0,1,...,N --experts <E>
```

`build_tiny_from_official.py:105-190` already renumbers layers contiguously and
rewrites `num_hidden_layers`, `compress_ratios`, `num_hash_layers` — no new
tooling needed.

**Also scale experts.** The tiny model uses 32 of the real 256, and MoE weights
dominate the checkpoint, so expert count matters as much as depth for memory and
compile.

Record per rung: compile wall-clock, instruction count, per-rank HBM, tokens/s.
The deliverable is a defensible extrapolation to 43 layers / 256 experts.

---

## Verification

**CPU, seconds** — after every step:

```bash
.venv/bin/python -m pytest test/unit/model/deepseek_v4/ \
    test/unit/test_deepseek_v4_small_context_buckets.py -q
```

Current result after the continuous-batching changes: **322 passed / 4 skipped**.
The full DeepSeek-V4 NKI simulator suite is **43 passed**.

**Device, per step** — greedy token comparison plus per-module activation diffing,
judged on median-row relative RMS and token equality (see the router-tie note):

`generate_tiny.py --capture-modules <comma-separated names>` limits capture to
the full-width boundaries relevant to a comparison. Compare two runs with
`tools/deepseek_v4/compare_tp_captures.py <reference> <actual>`; it fails when
capture sets/shapes differ or any module exceeds the median-row relative-RMS
threshold.

- **Step 1:** TP=8 post-change regression matches its reference exactly; TP=16
  Q8192 is deterministic and agrees through the first router-sensitive
  divergence; TP=32/EP=2 matches the TP=16 Q512 control exactly. TP=64/EP=4
  remains pending device availability.
- **Step 2:** compile wall-clock and instruction count fall with tokens unchanged.
- **Step 3:** complete for multi-request decode. Ragged batch=8 matches eight
  sequential runs exactly on TP=8 hardware; unit and simulator tests cover
  per-request pages, empty/new requests, candidate boundaries, and padded rows.
- **Step 4:** each rung generates coherent tokens; record the scaling curve.

### Known-good long-context invocation

Reaching 32K required three non-obvious settings:

```bash
--tensor-parallel-size 8 \
--max-model-len 32768 --max-num-batched-tokens 8192 \
--prefill-segment-buckets 8192 \      # must equal the segment size
--num-gpu-blocks-override 4096 \      # >=3232; tool default of 256 is far too small
--block-size 256 --max-num-seqs 1 --gpu-memory-utilization 0.9
```

- `MAX_MODEL_LEN_SINGLE_SHOT = 16384`, so **segmented prefill is mandatory above
  16K**; `max_num_batched_tokens` must be one of `{512, 1024, 2048, 4096, 8192}`.
- `generate_tiny.py` always passes `num_batched_tokens_buckets = [8, N]`, but
  segmented prefill requires it to equal the segment buckets exactly (`[8192]`).
- `generate_tiny.py:92` defaults `--num-gpu-blocks-override` to **256**, which
  throttles the KV budget to 0.08 GiB against the 1.01 GiB needed at 32K. vLLM
  itself sized the cache at 20,715 blocks, so HBM was never the constraint.

Both defaults break any run above ~2K context. Fixing them in the tool is a
worthwhile papercut cleanup (tracked below).

---

## Risks

- **TP>16 depends on EP, which has never run on this hardware.** If EP is broken
  the ceiling stays at 16 — which still fits the model at ~7.4 GB/rank, so this is
  a schedule risk rather than a blocker. Sequence EP validation early.
- **Compile cost may not fall enough.** If per-layer dedup is impossible and 43
  layers extrapolates to many hours, iterating on the real model becomes
  impractical and the ladder becomes the primary test vehicle. Decide after Step 2.
- **Mixed prefill/decode is still unsupported by the Neuron scheduler.** Decode
  continuously batches admitted requests, but prefills remain serialized. This
  is a throughput limitation rather than an indexer correctness blocker.
- **Disk.** The mid-depth fetch leaves ~116 GB free; NEFF artifacts and compile
  caches ran to hundreds of MB per configuration, and the ladder multiplies
  configurations.

## Deferred

- **Report 2, uninvestigated.** The `candidate_count == 1` / `q_count == 1`
  exposure at `nki_indexer.py:441-442`, flagged in
  `qwen3-5-tp8-device-bringup.md:125-127`. May be dead code now that decode pads
  Q1→Q8, but that is unconfirmed — and Step 3 touches exactly this code.
- **Decode throughput at long context.** 2.11 tok/s at 32K vs 9.68 at 2K on a
  288-token sequence: the static decode indexer scans full *compiled* capacity
  every step. Remedy is host-side page-count bucketing.
- **`generate_tiny.py` defaults** (`num_gpu_blocks_override=256`, `[8, N]` prefill
  buckets) break every config above ~2K.
- **Router ties** (~0.06% of rows). Inherent to BF16 top-k; not a correctness bug.
  Do not gate on exact token equality for long generations.
