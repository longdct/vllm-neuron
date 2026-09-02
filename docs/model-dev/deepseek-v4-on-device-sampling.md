# On-device sampling is wrong when the TP group spans Neuron devices

**Status:** fixed. At TP8 on trn2, on-device sampling now returns exactly the
host-sampled token ids, with async scheduling on and off. The fix is in
`functional/argmax.py` and `functional/topk.py`; see "The fix" below.

## Symptom

Any DeepSeek-V4 run with `--sampling-backend device` at TP8 dies one decode
step after the bad sample is taken:

```
ValueError: Token IDs out of range [0, 129280). Found min=739485477, max=739485477
```

The value is not random. Read as int32 it is the bit pattern of a small
float32 (`739485477` = `0x2c13a725` = `2.098e-12`), which is the magnitude of
this model's logits under dummy weights. The "token id" is float data read
through an int32 view.

It reproduces in BF16, so it has nothing to do with FP8.

## What it is

**The all-gather inside the distributed sampler only fills the slots belonging
to ranks on the same physical Neuron device. The rest are left uninitialized.**

Cores 12-19 are device 3 (cores 12-15) and device 4 (cores 16-19). Stamping
every NEFF output buffer before execution and reading back the sampled-token
output on all eight ranks gives:

| execution | ranks 0-3 | ranks 4-7 | agree? |
|---|---|---|---|
| 0 (prefill) | 63359 | 120936 | no |
| 1 | 6965 | 128613 | no |
| 2 | 63359 | 120936 | no |
| 3 | 6965 | 128613 | no |
| 4 (decode) | 739485477 | 726280733 | no |

Distributed argmax is documented to return the same value on every rank. It
never does here, and the split is exactly the device boundary. The in-range
values give the mechanism away: with a 16160-wide vocabulary shard, every
value ranks 0-3 produce lies in shards 0-3 and every value ranks 4-7 produce
lies in shards 4-7. Each half is doing argmax over its **own device's four
shards only**.

The gathered buffer is still eight slots wide, so the four slots belonging to
the other device hold whatever was in that memory. While that garbage stays
below the real maximum the result is merely wrong-but-plausible — a valid token
id, silently computed from half the vocabulary. When a garbage slot finally
wins the max, the index that comes back with it is unwritten memory, and that
is the out-of-range id. The failure is the visible tail of a defect that is
otherwise silent.

## What it is not

Each of these was tested and eliminated, in this order:

- **Not FP8.** Reproduces in BF16.
- **Not async scheduling.** `--sampling-backend device` also switched on
  `async_scheduling`; separating them (`--async-scheduling off`) fails
  identically. The two had never been varied independently.
- **Not the sampler.** `tools/deepseek_v4/probe_device_sampling.py` compiles
  the sampler alone and returns the exact expected tokens on device, on both
  the all-greedy and the generic top-k/top-p paths.
- **Not `torch.max`'s index output.** Correct on device, checked directly.
- **Not an unwritten output buffer.** Stamping every allocated NEFF output with
  a sentinel shows the sampled-token output *is* written every execution.
- **Not int64 index packing.** torch-neuronx defaults to
  `TORCH_NEURONX_INT64_MODE=packed` (int64 stored as two int32 lanes), which
  makes gathering an int64 index tensor look suspect. Narrowing the gathered
  indices to int32 changes the result not at all — bit-identical.

## The fix

Both distributed reductions now **gather the shards and reduce locally**,
instead of exchanging per-rank maxima and merging them.

The old algorithm is the cheaper one -- it moves `[batch, 1]` per rank instead
of `[batch, vocab_shard]` -- and that is exactly why it was wrong: the payload
was small enough to hit the defect. Gathering the shard is the same operation
`nn/cpl.py` performs on every host-sampled decode at the same TP degree, where
it is correct.

The cost is one vocabulary-width gather per sampling call (~517 KB at batch 1
for a 129280 vocabulary). On-device sampling still avoids the device-to-host
round trip, which is its main benefit.

Verified at depth 3, TP8, cores 12-19, dummy weights, greedy:

| configuration | tokens |
|---|---|
| host sampling (reference) | `[4868, 508, 4868, 508]` |
| on-device, async off | `[4868, 508, 4868, 508]` |
| on-device, async on | `[4868, 508, 4868, 508]` |

Exact equality is the right gate here, not "finite output": greedy argmax over
identical logits has no legitimate freedom to differ.

## Why padding was rejected

The obvious cheaper fix is to keep the exchange-maxima algorithm and pad the
tiny payload up to a width that gathers correctly. Measured, the width has to
be far larger than any useful `k`:

| gathered width | result at TP8 |
|---|---|
| 1 (the original) | wrong |
| 128 | wrong |
| 1024 | wrong |
| 16160 (one vocab shard) | correct |

Anything wide enough to work costs about as much as gathering the shard
outright, so padding buys nothing and leaves the code depending on an
undocumented threshold that could move.

## The trap when fixing it

`vllm_neuron/functional/argmax.py` and `topk.py` gather through
`torch.distributed._functional_collectives.all_gather_tensor`. Rewriting that
to the ordinary `dist.all_gather_into_tensor` — the formulation vLLM's own
device communicator uses — **changes nothing**, and the output is again
bit-identical. Dynamo remaps the `dist.*` collectives onto the same functional
collectives during tracing (`_functional_collectives.py`,
`traceable_collective_remaps`), so the two spellings compile to the same graph.
A fix has to change the collective that actually executes, not the one written
in Python.

Gathering on dim 0 instead of dim 1 does not help either, and neither does
narrowing the gathered indices to int32 -- both produce bit-identical wrong
output. Only the payload width matters.

Note also that `functional/collectives/all_gather_v.py` asserts a replica group
size of exactly 4, with a TODO to extend it. Group-size-4 is a real constraint
elsewhere in this collectives layer.

The underlying collective defect is not fixed by any of this: a narrow
all-gather across a TP group spanning Neuron devices still returns a
partially-uninitialized buffer, silently. That is worth reporting upstream --
any caller gathering a small tensor across such a group is exposed.

## Reproducing

`generate_tiny.py` now takes `--sampling-backend {cpu,device}` and
`--async-scheduling {on,off}` as independent switches, and both runners fail
loudly on an out-of-range generated token — the absence of that check is why
this went unnoticed. On the old tree the model had no dummy-weight initializer,
so an unwritten buffer read back as zero, and a whole depth-8 benchmark
"passed" while very possibly emitting token 0 every step.

Depth-3, dummy weights, greedy, seed 0:

- TP4 on one device (`NEURON_VISIBLE_DEVICES=12-15`): all ranks agree, and the
  device tokens `[4868, 508, 4868, 508]` match the CPU-sampled reference
  exactly. On-device sampling is correct here.
- TP8 across two devices (`12-19`): fails as above.

Timing numbers from earlier on-device runs are not invalidated — the decode
graph ran to completion with the right shapes each step. The *tokens* from
those runs are.
