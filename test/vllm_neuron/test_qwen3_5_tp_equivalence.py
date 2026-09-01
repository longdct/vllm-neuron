# SPDX-License-Identifier: Apache-2.0
"""TP=1 vs TP>1 forward equivalence for the Qwen3.5 decoder.

Why this file exists
--------------------
Nothing anywhere ran this model -- or its decoder layer, attention, MLP or
Gated DeltaNet -- at two tensor-parallel degrees and compared the results.
``test_qwen3_5_gdn_tp.py`` comes closest and still leaves ``world_size == 1``,
so **no collective ever fires**: the reduction is simulated by
``torch.stack(outputs).sum(0)`` in the test body, and ``forward_paged``'s
gather/scatter block is never entered. It also hand-patches ``layer.rank``,
which is precisely the failure DeepSeek recorded in its own suite -- a test
that supplies the rank itself stays green while the value the production code
computes is always 0.

That gap is how a KV-head mis-pairing survived as far as the 27B: at 24 Q / 4
KV heads and tp=16 the loader handed ranks 3, 6, 7, 9, 10 and 11 another GQA
group's keys, silently and with plausible output.

So this suite:

* runs **one thread per rank**, so ``all_gather`` / ``reduce_scatter`` /
  ``all_reduce`` really execute and really synchronize across ranks;
* takes every rank from the **production** ``resolve_tp_context``, patched in
  both ``model`` and ``gated_deltanet``, never from the test's loop variable;
* shards weights through each parameter's **own** production loader, so a
  partition error surfaces here as a numeric difference;
* asserts in every case that the collectives actually fired, and that a
  deliberately broken reduction *fails* the comparison.

Scope limit, stated rather than hidden
--------------------------------------
``VocabDimShardedEmbedding`` and ``ColumnParallelLinear`` size themselves from
global ``torch.distributed`` state (``dist.is_initialized()``), not from the TP
context, and their collectives are ``torch.distributed`` calls rather than
methods on the group object. A fake group therefore cannot drive them: with no
process group they behave as tp=1 whatever the context says. Embedding scatter
and the LM head are consequently out of scope here and stay covered on device.
Everything from the first decoder layer to the backbone's final all-gather is
in scope, which is where the partitioned arithmetic lives.
"""

import contextlib
import threading

import pytest

torch = pytest.importorskip("torch")

from vllm_neuron.model.qwen3_5 import gated_deltanet as gdn_mod  # noqa: E402
from vllm_neuron.model.qwen3_5 import model as model_mod  # noqa: E402
from vllm_neuron.model.qwen3_5.model import (  # noqa: E402
    Qwen3_5MLP,
    Qwen3_5TextForCausalLM,
    attention_layer_name,
    linear_layer_name,
)
from vllm_neuron.model.qwen3_5.parallel import (  # noqa: E402
    TPContext,
    resolve_sharding,
)

from vllm_neuron.utils.weight_loader import SliceView  # noqa: E402

from .test_qwen3_5_model_parity import (  # noqa: E402
    BLOCK_SIZE,
    NUM_BLOCKS,
    _config,
    _hf_config,
    _metadata,
    hf_modeling,
)

WORLD = 2
TOKENS = 8

# 4 Q / 2 KV and 2 key / 4 value heads, so the degrees are not interchangeable:
#   tp=2  splits every head count evenly -- the ordinary case;
#   tp=4  exhausts the key heads and falls back to splitting each value head's
#         *dimension*, where conv_dim_per_rank is 48 rather than 128 // 4 = 32;
#   tp=8  additionally pads 4 query heads up to 8, leaving ranks 4-7 holding
#         nothing but padding, which must contribute exactly zero.
# The last two are the shapes the 27B reaches at tp=16 and tp=32 and that no
# device run has ever executed.
WORLDS = (2, 4, 8)


# ---------------------------------------------------------------------------
# A tensor-parallel group that actually synchronizes
# ---------------------------------------------------------------------------

_local = threading.local()


class _FakeTPGroup:
    """One thread per rank, exchanging through a reusable barrier.

    Implements the three methods the model calls on ``self.tp_group``. Every
    collective is a real rendezvous: a rank that reaches one alone blocks, so a
    partition that fires collectives on unequal paths deadlocks rather than
    quietly producing a plausible number.

    All three return a *new* tensor and never mutate the input, which is the
    contract vLLM documents ("we always make the all-reduce operation
    out-of-place", ``parallel_state.py``). A caller that drops the result --
    as three sites in this model used to -- fails here.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self.device_group = None
        self._barrier = threading.Barrier(world_size)
        self._slots = [None] * world_size
        self._lock = threading.Lock()
        self.calls = {"all_gather": 0, "reduce_scatter": 0, "all_reduce": 0}

    @property
    def rank_in_group(self):
        return _local.rank

    def abort(self):
        self._barrier.abort()

    def _exchange(self, tensor):
        self._slots[_local.rank] = tensor.detach().clone()
        self._barrier.wait()
        gathered = list(self._slots)
        self._barrier.wait()
        return gathered

    def _count(self, name):
        with self._lock:
            self.calls[name] += 1

    def all_gather(self, tensor, dim=0):
        self._count("all_gather")
        return torch.cat(self._exchange(tensor), dim=dim)

    def all_reduce(self, tensor):
        self._count("all_reduce")
        return torch.stack(self._exchange(tensor), dim=0).sum(dim=0)

    def reduce_scatter(self, tensor, dim=0):
        self._count("reduce_scatter")
        total = torch.stack(self._exchange(tensor), dim=0).sum(dim=0)
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, _local.rank * width, width).contiguous()


class _RankZeroOnlyGroup(_FakeTPGroup):
    """Anti-vacuity: a reduction that keeps only rank 0's partial.

    Shaped exactly like the real thing -- same signatures, same output shapes,
    same collective counts -- and arithmetically wrong. Any test that still
    passes with this substituted was not testing the reduction.
    """

    def all_reduce(self, tensor):
        self._count("all_reduce")
        return self._exchange(tensor)[0] * self.world_size

    def reduce_scatter(self, tensor, dim=0):
        self._count("reduce_scatter")
        total = self._exchange(tensor)[0] * self.world_size
        width = total.shape[dim] // self.world_size
        return total.narrow(dim, _local.rank * width, width).contiguous()


@contextlib.contextmanager
def _tp_context(group):
    """Make the *production* resolver report this thread's rank.

    Patched in both modules that call it. Nothing in a test body ever assigns a
    rank onto a module: every ``self.rank`` / ``self.world_size`` in the model
    comes from here, so a module that reads the rank from the wrong place is
    caught rather than accommodated.
    """

    def resolve():
        return TPContext(group, group.world_size, getattr(_local, "rank", 0))

    saved = (model_mod.resolve_tp_context, gdn_mod.resolve_tp_context)
    model_mod.resolve_tp_context = resolve
    gdn_mod.resolve_tp_context = resolve
    try:
        yield
    finally:
        model_mod.resolve_tp_context, gdn_mod.resolve_tp_context = saved


def _run_ranks(group, body):
    """Run ``body(rank)`` on one thread per rank; return results in rank order."""
    world = group.world_size
    results = [None] * world
    errors = [None] * world

    def target(rank):
        _local.rank = rank
        try:
            results[rank] = body(rank)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[rank] = exc
            group.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(world)]
    with _tp_context(group):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

    for thread in threads:
        assert not thread.is_alive(), "a rank deadlocked on a collective"
    for error in errors:
        if error is not None:
            raise error
    return results


# ---------------------------------------------------------------------------
# Weights: sharded by the production loaders, never by hand
# ---------------------------------------------------------------------------

_FUSED_QKV = "qkv_proj_weight"
_PROJ = "_proj_weight"


def _source_keys(name):
    """Map one of our parameter names onto the HF keys that feed it.

    Our fused projections spell the parameter ``..._proj_weight`` where the
    checkpoint has a submodule, ``..._proj.weight``. Handled by suffix so the
    same mapping serves a bare ``Qwen3_5MLP`` and a whole model, whose
    parameters differ only by prefix.
    """
    stem = name[len("model.") :] if name.startswith("model.") else name
    if stem.endswith(_FUSED_QKV):
        head = stem[: -len(_FUSED_QKV)]
        return [f"{head}{p}_proj.weight" for p in ("q", "k", "v")]
    if stem.endswith(_PROJ):
        return [stem[: -len(_PROJ)] + "_proj.weight"]
    return [stem]


def _shard_state_dict(module, source, rank):
    """Build ``rank``'s parameters by invoking each one's own loader.

    The loaders are the production objects the modules attached to themselves,
    carrying the resolved policy -- including the ``+1`` norm fold and the
    per-head query/gate fusion. Nothing here slices a tensor: if the partition
    is wrong, it is wrong here in exactly the way it is wrong in the engine.

    ``SliceView`` is the production adapter, not a test double: it presents the
    ``get_shape()`` + ``__getitem__`` checkpoint-slice interface over a plain
    tensor, so the loaders run against what they see in the engine.
    """
    target = {}
    for name, param in module.named_parameters():
        if name.startswith("lm_head"):
            continue  # sizes from global dist state; see the module docstring
        keys = _source_keys(name)
        loader = getattr(param, "weight_loader", None)
        assert loader is not None, f"{name} carries no production weight loader"
        loaded = loader.load([SliceView(source[k]) for k in keys], rank=rank)
        assert loaded.shape == param.shape, (
            f"{name}: loader produced {tuple(loaded.shape)} for a "
            f"{tuple(param.shape)} parameter at rank {rank}"
        )
        target[name] = loaded.detach().clone().float()
    return target


def _caches(config, policy):
    """State tensors at this rank's widths, not the unsharded ones."""
    caches = {}
    for i, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            shape = (NUM_BLOCKS, policy.kv_heads_per_rank, BLOCK_SIZE, config.head_dim)
            caches[attention_layer_name(i)] = [torch.zeros(shape), torch.zeros(shape)]
        else:
            caches[linear_layer_name(i)] = [
                torch.zeros(
                    NUM_BLOCKS,
                    policy.conv_dim_per_rank,
                    config.linear_conv_kernel_dim - 1,
                ),
                torch.zeros(
                    NUM_BLOCKS,
                    policy.v_heads_per_rank,
                    config.linear_key_head_dim,
                    policy.v_dim_per_rank,
                ),
            ]
    return caches


@pytest.fixture(scope="module")
def source():
    """One unsharded weight set, in HuggingFace layout, shared by every degree."""
    config = _config()
    torch.manual_seed(0)
    hf = hf_modeling.Qwen3_5TextModel(_hf_config(config)).eval().float()
    return {k: v.detach().clone() for k, v in hf.named_parameters()}


def _build(config, source, rank, world):
    """A loaded, cache-bound model for ``rank``. Must run inside its thread."""
    model = Qwen3_5TextForCausalLM(config).eval().float()
    assert model.world_size == world, (
        f"model resolved world_size={model.world_size}, expected {world}: the "
        "production resolver is not driving this build"
    )
    missing, unexpected = model.load_state_dict(
        _shard_state_dict(model, source, rank), strict=False
    )
    assert not unexpected, unexpected
    assert all(n.startswith("lm_head") for n in missing), missing
    model.bind_kv_cache(_caches(config, model.policy))
    return model


def _run_stack(model, hidden, positions, metadata, is_prefill):
    """The backbone's tail: layers, final norm, and the sequence-parallel gather.

    Reproduced here rather than called because ``Qwen3_5TextModel.forward``
    enters through the embedding, which a fake group cannot shard.
    """
    embeddings = model.model.rotary_emb(
        positions, device=hidden.device, dtype=hidden.dtype
    )
    for layer in model.model.layers:
        hidden = layer(hidden, positions, embeddings, metadata)
    hidden = model.model.norm(hidden)
    if is_prefill and model.world_size > 1:
        hidden = model.tp_group.all_gather(hidden, dim=0)
    return hidden


def _reference(config, source):
    """The tp=1 model, built through the genuine unpatched resolver."""
    return _build(config, source, rank=0, world=1)


def _first_full_attention_layer(config):
    return config.layer_types.index("full_attention")


# ---------------------------------------------------------------------------
# The harness itself
# ---------------------------------------------------------------------------


def test_the_fake_group_reduces_across_ranks():
    """Guard the guard: a group that ignored its peers would pass everything."""
    group = _FakeTPGroup(WORLD)
    values = _run_ranks(
        group, lambda rank: group.all_reduce(torch.full((3,), rank + 1.0))
    )

    for value in values:
        torch.testing.assert_close(value, torch.full((3,), 3.0))
    assert group.calls["all_reduce"] == WORLD


def test_the_fake_group_scatters_each_rank_its_own_slice():
    group = _FakeTPGroup(WORLD)

    def body(rank):
        return group.reduce_scatter(torch.arange(8.0) * (rank + 1), dim=0)

    slices = _run_ranks(group, body)
    # Both ranks contribute, so the total is 3x; rank r keeps its own half.
    torch.testing.assert_close(slices[0], torch.arange(0.0, 4.0) * 3)
    torch.testing.assert_close(slices[1], torch.arange(4.0, 8.0) * 3)


def test_the_broken_group_is_detectably_wrong():
    """The anti-vacuity double is shaped right and arithmetically wrong."""
    group = _RankZeroOnlyGroup(WORLD)
    values = _run_ranks(
        group, lambda rank: group.all_reduce(torch.full((3,), rank + 1.0))
    )

    for value in values:
        assert not torch.allclose(value, torch.full((3,), 3.0))


@pytest.mark.parametrize("world", WORLDS)
def test_ranks_come_from_the_production_resolver(world):
    """No test body assigns a rank; the model must derive it itself.

    This is the DeepSeek lesson: their loader test passed ``expert_tp_rank`` in
    by hand and stayed green while the production value was always 0.
    """
    config = _config()
    group = _FakeTPGroup(world)

    def body(rank):
        mlp = Qwen3_5MLP(config, resolve_sharding(config, world))
        return mlp.world_size, model_mod.resolve_tp_context().rank

    seen = _run_ranks(group, body)
    assert [w for w, _ in seen] == [world] * world
    assert [r for _, r in seen] == list(range(world))


# ---------------------------------------------------------------------------
# Qwen3_5MLP -- no test existed for this module at any degree
# ---------------------------------------------------------------------------


def _mlp_source(source):
    """The layer-0 MLP weights, keyed as a bare Qwen3_5MLP expects them."""
    return {
        "gate_proj.weight": source["layers.0.mlp.gate_proj.weight"],
        "up_proj.weight": source["layers.0.mlp.up_proj.weight"],
        "down_proj.weight": source["layers.0.mlp.down_proj.weight"],
    }


def _mlp_case(source, group, is_prefill):
    config = _config()
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(TOKENS, config.hidden_size, generator=generator)
    weights = _mlp_source(source)

    reference = Qwen3_5MLP(config, resolve_sharding(config, 1)).eval().float()
    reference.load_state_dict(_shard_state_dict(reference, weights, rank=0))
    with torch.no_grad():
        expected = reference(hidden, is_prefill=is_prefill)

    def body(rank):
        policy = resolve_sharding(config, group.world_size)
        mlp = Qwen3_5MLP(config, policy).eval().float()
        mlp.load_state_dict(_shard_state_dict(mlp, weights, rank=rank))
        local = hidden.chunk(group.world_size, dim=0)[rank] if is_prefill else hidden
        with torch.no_grad():
            return mlp(local, is_prefill=is_prefill)

    return expected, _run_ranks(group, body)


@pytest.mark.parametrize("world", WORLDS)
def test_mlp_prefill_matches_tp1(source, world):
    """Sequence-parallel MLP: scattered in, reduce-scattered out."""
    group = _FakeTPGroup(world)
    expected, shards = _mlp_case(source, group, is_prefill=True)

    torch.testing.assert_close(torch.cat(shards, dim=0), expected, rtol=1e-5, atol=1e-5)
    assert group.calls["all_gather"] == world
    assert group.calls["reduce_scatter"] == world
    assert group.calls["all_reduce"] == 0


@pytest.mark.parametrize("world", WORLDS)
def test_mlp_decode_matches_tp1(source, world):
    """Replicated MLP: every rank must hold the whole, fully reduced answer."""
    group = _FakeTPGroup(world)
    expected, replicas = _mlp_case(source, group, is_prefill=False)

    for replica in replicas:
        torch.testing.assert_close(replica, expected, rtol=1e-5, atol=1e-5)
    assert group.calls["all_reduce"] == world
    assert group.calls["all_gather"] == 0


@pytest.mark.parametrize("is_prefill", [True, False])
def test_mlp_fails_when_the_reduction_drops_a_rank(source, is_prefill):
    """The reduction is load-bearing: break it and the comparison must fail."""
    group = _RankZeroOnlyGroup(WORLD)
    expected, outputs = _mlp_case(source, group, is_prefill=is_prefill)
    actual = torch.cat(outputs, dim=0) if is_prefill else outputs[0]

    assert not torch.allclose(actual, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Decoder layers, both kinds, and the whole stack
# ---------------------------------------------------------------------------


def _prefill_case(source, group):
    config = _config()
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(TOKENS, config.hidden_size, generator=generator)
    positions = torch.arange(TOKENS, dtype=torch.int32)
    metadata = _metadata(config, TOKENS)

    reference = _reference(config, source)
    with torch.no_grad():
        expected = _run_stack(reference, hidden, positions, metadata, is_prefill=True)

    def body(rank):
        model = _build(config, source, rank, group.world_size)
        local = hidden.chunk(group.world_size, dim=0)[rank]
        with torch.no_grad():
            return _run_stack(model, local, positions, metadata, is_prefill=True)

    return expected, _run_ranks(group, body)


@pytest.mark.parametrize("world", WORLDS)
def test_prefill_stack_matches_tp1(source, world):
    """The sequence-parallel path: scatter, run the hybrid stack, gather.

    Reachable only on device before this test existed.
    """
    group = _FakeTPGroup(world)
    expected, gathered = _prefill_case(source, group)

    # Every rank ends the backbone holding the same all-gathered result.
    for actual in gathered:
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    # Both layer kinds and the MLP gather on entry and scatter on exit.
    assert group.calls["all_gather"] > 0
    assert group.calls["reduce_scatter"] > 0


def test_prefill_stack_fails_when_the_reduction_drops_a_rank(source):
    group = _RankZeroOnlyGroup(WORLD)
    expected, gathered = _prefill_case(source, group)

    assert not torch.allclose(gathered[0], expected, rtol=1e-3, atol=1e-3)


def _decode_case(source, group):
    """Prefill to fill both cache groups, then a single decode step."""
    config = _config()
    generator = torch.Generator().manual_seed(13)
    prompt = torch.randn(TOKENS, config.hidden_size, generator=generator)
    step = torch.randn(1, config.hidden_size, generator=generator)

    prefill_meta = _metadata(config, TOKENS)
    decode_meta = _metadata(
        config, 1, cached_seq_len=TOKENS, is_decode=True, start=TOKENS
    )
    prefill_pos = torch.arange(TOKENS, dtype=torch.int32)
    decode_pos = torch.tensor([TOKENS], dtype=torch.int32)

    reference = _reference(config, source)
    with torch.no_grad():
        _run_stack(reference, prompt, prefill_pos, prefill_meta, is_prefill=True)
        expected = _run_stack(reference, step, decode_pos, decode_meta, is_prefill=False)

    def body(rank):
        model = _build(config, source, rank, group.world_size)
        local = prompt.chunk(group.world_size, dim=0)[rank]
        with torch.no_grad():
            _run_stack(model, local, prefill_pos, prefill_meta, is_prefill=True)
            # Decode is replicated: every rank holds the whole step.
            return _run_stack(model, step, decode_pos, decode_meta, is_prefill=False)

    return expected, _run_ranks(group, body)


@pytest.mark.parametrize("world", WORLDS)
def test_decode_after_prefill_matches_tp1(source, world):
    """Carries paged KV *and* Gated DeltaNet state across the seam.

    A partition that is right for a fresh prefill and wrong for state written
    by a previous pass fails here and nowhere else.
    """
    group = _FakeTPGroup(world)
    expected, replicas = _decode_case(source, group)

    for replica in replicas:
        torch.testing.assert_close(replica, expected, rtol=2e-5, atol=2e-5)
    assert group.calls["all_reduce"] > 0


def test_decode_fails_when_the_reduction_drops_a_rank(source):
    group = _RankZeroOnlyGroup(WORLD)
    expected, replicas = _decode_case(source, group)

    assert not torch.allclose(replicas[0], expected, rtol=1e-3, atol=1e-3)


def test_every_rank_holds_a_different_shard(source):
    """Guard against a 'sharding' that replicates.

    If each rank silently loaded the same weights, every equivalence test above
    would still pass -- the reduction would just be summing duplicates -- so
    require the partitions to actually differ.
    """
    config = _config()
    group = _FakeTPGroup(WORLD)
    gdn_index = config.linear_layer_indices[0]
    attn_index = _first_full_attention_layer(config)

    def body(rank):
        model = _build(config, source, rank, group.world_size)
        layers = model.model.layers
        return (
            layers[gdn_index].linear_attn.in_proj_qkv.weight.detach().clone(),
            layers[attn_index].self_attn.qkv_proj_weight.detach().clone(),
            layers[0].mlp.gate_proj_weight.detach().clone(),
        )

    shards = _run_ranks(group, body)
    for index, what in enumerate(("gdn in_proj_qkv", "attention qkv", "mlp gate_proj")):
        assert not torch.allclose(shards[0][index], shards[1][index]), (
            f"{what} is identical on both ranks -- it is replicated, not sharded"
        )
