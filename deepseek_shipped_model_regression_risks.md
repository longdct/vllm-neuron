# Shipped-model regression run — failure modes and phase impact

Companion to `deepseek_actual_implement_plan.md`. That plan's P0.5 development gate ends on one
clause that cannot run off-hardware: **GPT-OSS and Qwen3-VL serve at T3 with acceptable logit drift
against the P0.1 baseline.** This document is what to consult when that run finally happens and
something goes wrong.

Its purpose is triage, not reassurance. For each failure mode: what it looks like, how likely it is,
what in the 0.21→0.26 move causes it, and — the part that matters for scheduling — **which later
phases inherit the problem.** Several of these failures would invalidate DeepSeek work that had
already been declared done, and knowing which ones in advance is the difference between a bad day
and a bad month.

Every entry is grounded in the working tree at the time of writing, not inferred from the diff.

---

## The meta-risk: there is no baseline to compare against

**R0 — the P0.1 pre-upgrade baseline was never captured. Severity: blocking.**

The plan requires baseline logits for GPT-OSS and Qwen3-VL captured *before* the pin moved, and is
explicit that this "must exist before the pin moves or it cannot be created afterwards." The pin
moved at commit `5ce9f8f`. No baseline artifacts exist anywhere in the tree.

**Consequence: as specified, the T3 clause is currently unrunnable.** "Acceptable drift" has no
referent. Recovering it means checking out `be0def6`, installing vLLM 0.21, and running on Trn2
*first* — so the regression run is two Trainium sessions, not one, and they must bracket the pin.

Two things follow, and both are scheduling facts rather than technical ones:

- Any plan that budgets one Trn2 session for P0 is wrong by a factor of two.
- If the 0.21 side is skipped and drift is instead judged against *expectation*, the run stops being
  a regression test and becomes a smoke test. That is still worth doing — it catches crashes, OOM,
  and gross corruption — but it must not be recorded as satisfying P0.5, because it cannot detect
  the silent, small-magnitude drift the clause exists to catch.

A tempting shortcut worth naming so it can be rejected: comparing the 0.26 device run against a
**CPU-mode** run instead. This does not work. CPU mode substitutes PyTorch fallbacks for NKI kernels
(`docs/model-dev/cpu-development.md`), so the two numbers are produced by different code. A
disagreement would be uninterpretable, and — worse — an *agreement* would be actively misleading.

---

## Failure modes, most consequential first

### R1 — Block tables undersized for encoder length

**Severity: mitigated by rejection.** Not reachable by any currently supported model.

`build_input_batch_group_params` sizes block tables from `self.max_model_len`. Upstream's equivalent
(`gpu_model_runner.may_reinitialize_input_batch`) uses `max(self.max_model_len, self.max_encoder_len)`,
where `max_encoder_len` is non-zero only when `model_config.is_encoder_decoder`.

The Neuron runner has **no `max_encoder_len` concept at all** — the attribute does not exist. All four
registered models (`LlamaForCausalLM`, `GptOssForCausalLM`, `Eagle3LlamaForCausalLM`,
`Qwen3VLForConditionalGeneration`) are decoder-only; Qwen3-VL is multimodal but not encoder-decoder,
so it does not trip this.

**What it would look like:** not a crash. Block table rows too short for the encoder span, producing
reads past the intended region — garbled output on long inputs, correct output on short ones.

**Mitigation:** `build_input_batch_group_params` now raises for `is_encoder_decoder` before block
tables are constructed. Supporting such a model requires adding `max_encoder_len` to the runner and
matching upstream's `max(max_model_len, max_encoder_len)` sizing first.

### R2 — `kernel_block_sizes` assumed equal to `block_sizes`

**Severity: locally verified backend invariant.**

Upstream treats `kernel_block_sizes` as **backend-determined** — it is a *parameter* to
`may_reinitialize_input_batch`, supplied by the attention backend, precisely so a kernel can address
a sub-block. The Neuron derivation sets it equal to `block_sizes`.

This is **not a new regression**: the pre-0.26 Neuron code did the same thing. A repository-wide
backend audit also found no sub-block addressing contract: QKV cache scatter and segmented attention
gather both divide/modulo by the page `block_size` directly. Equality is therefore the only layout
the current backend can express. Keep the invariant guarded; a future sub-block kernel must add an
explicit backend-provided value rather than silently changing this derivation.

### R3 — `max_num_blocks_per_req` interacts with decode context parallelism

**Severity: medium, scoped to DCP deployments.**

This argument did not exist at 0.21; it is now required, and its value comes from upstream's per-spec
method. For `SlidingWindowSpec` that method divides by `decode_context_parallel_size`:

```
cdiv(max_len, self.block_size * kv_shard_count)
```

The Neuron DCP path has its own block-table expectations, and this is the first release where
upstream's DCP-aware sizing feeds it. A mismatch produces a block table row length that disagrees
with what the Neuron kernel indexes — again silent, again data-dependent.

**Coupling worth noting:** DCP on Neuron requires `NeuronNixlConnector` (enforced in `platform.py`),
which R4 now rejects outright. So **DCP configurations cannot currently be tested at all.** R3 is
therefore untestable until R4 is resolved — a dependency that is easy to miss when planning the run.

### R4 — Disaggregated inference now fails hard, by design

**Severity: intended behavioral regression.**

Any shipped 1P1D deployment using `NeuronNixlConnector` now fails at startup with
`UnsupportedDIConfigError`. This is the plan's P0.4 decision working as intended — loud beats silent —
but it *is* a regression against 0.21 for real deployments, and it will surface in the shipped-model
run as a failure if that run covers DI topologies.

**Do not "fix" this by weakening the rejection.** The correct responses are either to restore the
connector (the pull/push split; plan P0.5 release gate) or to scope the run to non-DI topologies and
record DI as explicitly unsupported in this artifact.

### R5 — Transformers ceiling may break existing environments

**Severity: low, immediate and loud.**

`requirements/core.txt` and `requirements/test.txt` now pin `transformers>=5.5.3,<5.16`, narrowed from
`<6.0.0` to the version actually exercised (5.15.0). An environment that previously resolved to
something newer will now fail to install.

This fails at dependency resolution — noisy, immediate, trivially diagnosed. It is listed only so it
is not mistaken for a model-level regression when it appears in CI.

### R6 — Port guards check names, not meaning

**Severity: structural, ongoing.**

`TestPortedUpstreamSurfaces` asserts that the fields the two ported bodies read still *exist* on
`CachedRequestData`, `CachedRequestState`, `Request` and `SchedulerOutput`. It cannot assert that they
still *mean* the same thing.

A field that keeps its name while changing units, base, or lifecycle — `num_computed_tokens` starting
to include speculative tokens, say — passes every guard and silently corrupts scheduling. This is
precisely the class of drift the shipped-model run exists to catch, which is why R0 (no baseline)
compounds so badly: the guards cover the loud half, and the run that covers the quiet half cannot
currently be scored.

### R7 — `slot_mapping_modes` — ruled out

**Severity: none. Verified equivalent.**

Newly passed explicitly where 0.21 passed nothing. `MultiGroupBlockTable` defaults `None` to
`[SlotMappingMode.TOKEN_TO_KV_SLOT] * len(block_sizes)`, which is exactly what the derivation produces
for every non-Mamba spec — and no Neuron cache group is Mamba.

Recorded deliberately: knowing which changes are *not* suspects is as useful during triage as knowing
which are.

---

## How each failure propagates into later phases

The column that matters is the rightmost one. A failure that only costs a fix is cheap; a failure
that invalidates completed work is what wrecks a schedule.

| Risk | P1 cache | P2 512-d MLA | P3 tiny model | P4 oracles | P6 e2e logits | P7/P8 memory + M1 | Invalidates finished work? |
|---|---|---|---|---|---|---|---|
| **R0** no baseline | — | — | — | — | **yes** | **yes** | **No, but it un-scores them.** P6 and P8 both compare against references whose credibility rests on the runtime being known-good. Neither result becomes *wrong*; both become unattributable. |
| **R1** encoder length | — | — | — | — | — | — | No. Latent until an encoder-decoder model is added. |
| **R2** kernel block sizes | **high** | medium | medium | — | — | — | **Potentially yes.** P1 defines c4/c128 layouts on this assumption. If a Neuron backend needs a kernel block size ≠ page block size, P1.3's layouts and P1.5's binding are both rederived. Settle it *in* P1. |
| **R3** DCP block sizing | medium | — | — | — | — | high | No, but it blocks TP/SP/EP validation in P6 and P8, which both require DCP-adjacent topologies to hold. |
| **R4** DI rejected | low | — | — | — | — | **high** | No. But P8's exit gate requires unsupported combinations to fail at config time — R4 *is* that behavior, so P8 depends on it staying in place. Removing it to unblock R3 would breach P8's gate. |
| **R5** transformers pin | — | — | **medium** | **high** | — | — | No. P3a.1 and P4's config oracle are defined against the Transformers config loader; a version change moves the oracle itself. Re-pin deliberately, never incidentally. |
| **R6** semantic drift | **high** | — | medium | — | **high** | **high** | **Yes, silently.** This is the one that can retroactively invalidate a green P6. Scheduling drift corrupts cache lifecycle, which is P1's entire subject — and P1's gate is exact discrete comparison, which would *pass* against a consistently-wrong scheduler. |
| **R7** slot mapping | — | — | — | — | — | — | No. Ruled out. |

### Reading the table

Three conclusions worth acting on:

**R6 is the expensive one.** It is the only entry that can invalidate work already marked done, and it
is the one the guards cannot catch. Its mitigation is not more unit tests — it is the baselined
shipped-model run, i.e. resolving R0. That is the concrete argument for spending the second Trainium
session rather than skipping it.

**R2 should be settled inside P1, not deferred.** It is cheap to answer now (does any Neuron attention
backend want a kernel block size different from the page block size?) and expensive to answer after
P1.3 and P1.5 are built on the assumption.

**R3 and R4 are entangled and should be sequenced together.** DCP requires the NIXL connector; the
connector is rejected; so DCP is untestable. Either restore the connector before validating DCP
sizing, or accept that both stay unvalidated until the P0.5 release gate — but do not plan them as
independent work items.

---

## If the run cannot be baselined

Should the second Trainium session prove unavailable, the honest fallback is to run the 0.26 side
only and record it as a **smoke test**, with these constraints stated in the artifact:

- It detects crashes, OOM, NEFF load failures and gross output corruption.
- It does **not** detect small-magnitude logit drift, which is the failure mode P0.5 names.
- P0.5's development gate stays **open**, and every downstream phase that cites it inherits that
  caveat — P6 and P8 in particular.

The plan's own instruction applies here without modification: *"Store pinned baseline logits and
report drift — top-token agreement alone hides real regressions."* A smoke test that reports
top-token agreement is not a weaker version of the gate. It is a different measurement.
