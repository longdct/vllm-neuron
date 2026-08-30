# SPDX-License-Identifier: Apache-2.0
"""
NKI kernels for the Qwen3.5 Gated DeltaNet.

The torch forms in ``gated_deltanet.py`` are the oracle; these are the device
path. Both live behind one dispatcher each, so a kernel that is unavailable
(older wheel, CPU-only environment) degrades to the reference rather than
failing to import -- the pattern in
``functional/attention/swa_fused.py``.

The causal convolution reuses ``nkilib``'s depthwise conv rather than being
hand-written: it ships with its own torch reference, which doubles as this
module's fallback and its test oracle.

Note on the conv and history. A zero conv state is *exactly* the reference's
left zero-padding, so there is only one code path: always concatenate the
state -- zeros for a fresh sequence -- and convolve with no padding. That
removes a branch on "is this a new sequence?", which on a traced graph would be
a data-dependent shape.

Note on testing these on CPU. ``wrap_nki`` here is ``torch_neuronx.nki_hop``'s,
matching every other kernel wrapper in this repo, and it requires real Neuron
tensors -- ``VLLM_NEURON_CPU_MODE=1`` does not give it a CPU fallback. So the
simulator tests drive ``nki.simulate(kernel[lnc])`` directly rather than going
through the dispatchers below, which is the same arrangement
``test_deepseek_v4_nki_simulator.py`` uses.
"""

import logging

import torch
import torch.nn.functional as F

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kernel availability
# ---------------------------------------------------------------------------

_wrapped_depthwise_conv1d = None
try:  # pragma: no cover - depends on the installed nkilib
    import nki
    from torch_neuronx.nki_hop import wrap_nki

    from nkilib.experimental.conv.depthwise_conv1d import (
        depthwise_conv1d_implicit_gemm,
    )

    _depthwise_conv1d_jit = nki.jit()(depthwise_conv1d_implicit_gemm)
    _wrapped_depthwise_conv1d = wrap_nki(_depthwise_conv1d_jit)
except Exception as exc:  # noqa: BLE001
    logger.debug(
        "nkilib depthwise_conv1d unavailable, Gated DeltaNet will use torch: %s", exc
    )


#: LNC grid. The depthwise conv shards on the channel dimension, and every
#: Qwen3.5 conv width is a multiple of 2, so LNC2 is always splittable.
_LNC = 2


def _can_use_conv_kernel(x: torch.Tensor, channels: int) -> bool:
    if _wrapped_depthwise_conv1d is None:
        return False
    if not can_run_kernel(x):
        return False
    # The kernel shards channels across LNC; an odd width cannot split.
    return channels % _LNC == 0


# ---------------------------------------------------------------------------
# Causal depthwise convolution
# ---------------------------------------------------------------------------


def _torch_causal_conv1d(extended, weight, activation):
    """Reference: convolve ``[state | tokens]`` with no padding."""
    channels = extended.shape[1]
    out = F.conv1d(
        extended.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=None,
        padding=0,
        groups=channels,
    )
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported conv activation {activation!r}")
    return out


def causal_conv1d_with_state(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    valid_len: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv over ``[state | tokens]``.

    Args:
        hidden_states: ``[B, C, T]`` new tokens.
        conv_state: ``[B, C, kernel - 1]`` trailing inputs from earlier steps.
            All zeros represents a fresh sequence, which is exactly the
            reference's left zero-padding -- so this is not a special case.
        weight: ``[C, kernel]`` depthwise filter.
        bias: must be None; Qwen3.5's conv1d carries no bias.
        valid_len: ``[B]`` count of leading real tokens per row, or None when
            every row is real. Only the returned state depends on it -- the
            output is computed for every column either way, because a padded
            column's output is discarded downstream and computing it costs
            nothing, whereas skipping it would need a data-dependent shape.

    Returns:
        ``(output, new_conv_state)`` with output covering the new tokens only.
    """
    if bias is not None:
        raise ValueError("Qwen3.5's Gated DeltaNet conv1d has no bias")

    kernel = weight.shape[-1]
    seq_len = hidden_states.shape[-1]
    channels = hidden_states.shape[1]

    # The conv runs at the activation dtype, so the state joins it there. The
    # cached state may be stored wider -- it shares a page with the fp32
    # recurrent accumulator, and both must carry the same dtype for the
    # backend to allow writing them back in place (see model.py::get_kv_spec).
    # Concatenating without this cast promotes the whole window to fp32 and the
    # kernel then rejects the pair: "nc_matmul: if one input is
    # tfloat32/float32, both must be. Got stationary=bfloat16, moving=float32".
    # This function already treats hidden_states.dtype as authoritative -- it
    # casts the result back to it on return.
    extended = torch.cat([conv_state.to(hidden_states.dtype), hidden_states], dim=-1)

    if _can_use_conv_kernel(hidden_states, channels):
        # nkilib wants [N, C, 1, W] and [C, 1, 1, S]; the state supplies the
        # left context, so no padding is requested.
        #
        # feature_group_count MUST equal the channel count. The kernel is
        # "depthwise" by validation, not by its default: leaving it at 1 raises
        # NCC_INKI016 ("feature_group_count must equal C"), which only surfaces
        # once a kernel actually runs -- so it is asserted by a simulator test
        # rather than trusted.
        out = _wrapped_depthwise_conv1d[_LNC](
            extended.unsqueeze(2).contiguous(),
            weight.reshape(channels, 1, 1, kernel).contiguous(),
            padding=((0, 0), (0, 0)),
            feature_group_count=channels,
        ).squeeze(2)
        if activation == "silu":
            out = F.silu(out)
        elif activation is not None:
            raise ValueError(f"Unsupported conv activation {activation!r}")
    else:
        out = _torch_causal_conv1d(extended, weight, activation)

    out = out[..., -seq_len:]
    new_state = (
        _trailing_window(extended, kernel - 1, valid_len)
        if kernel > 1
        else conv_state
    )
    return out.to(hidden_states.dtype), new_state


def _trailing_window(
    extended: torch.Tensor, width: int, valid_len: torch.Tensor | None
) -> torch.Tensor:
    """The ``width`` columns of ``extended`` ending at each row's last real token.

    ``extended`` is ``[state | tokens]``, so real token ``j`` sits at column
    ``width + j`` and the window ending at the last real token of a row with
    ``n`` real tokens is ``extended[..., n : n + width]``. With no padding
    ``n == seq_len`` and that is the plain trailing window.

    Padding makes the distinction load bearing. A bucketed prefill appends
    padding rows to every request, so the plain trailing window is a window
    over *padding*, and decode then resumes the convolution from tokens the
    prompt never contained.

    The offset is read from a tensor and the gather has a fixed width, so no
    Python ``int`` reaches a shape -- the discipline §4.2 of the plan requires
    and that cost DeepSeek-V4 nine consecutive Dynamo blockers.
    """
    if valid_len is None:
        return extended[..., -width:]
    # [B, width] absolute columns, then broadcast over the channel dim.
    offsets = valid_len.to(torch.long).reshape(-1, 1) + torch.arange(
        width, device=extended.device
    ).reshape(1, -1)
    index = offsets.unsqueeze(1).expand(-1, extended.shape[1], -1)
    return torch.gather(extended, 2, index)


def causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Fresh-sequence convolution: the stateful form with a zero state."""
    kernel = weight.shape[-1]
    state = torch.zeros(
        hidden_states.shape[0],
        hidden_states.shape[1],
        kernel - 1,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    out, _ = causal_conv1d_with_state(hidden_states, state, weight, bias, activation)
    return out


# ---------------------------------------------------------------------------
# Chunk scan: the sequential half of the delta rule
# ---------------------------------------------------------------------------
#
# Only the inter-chunk state recurrence is sequential. Everything else the
# chunked delta rule does -- the intra-chunk decay mask, the UT-transform
# inverse, ``attn @ v_beta``, ``k_cumdecay`` -- is independent per chunk and
# traces as ordinary batched tensor ops that do not unroll. So the kernel's job
# is exactly the loop, and nothing more:
#
#   v_new_i     = v_i - k_cumdecay_i @ S
#   out_i       = (q_i * exp(g_i)) @ S + attn_i @ v_new_i
#   S           = S * exp(g_last_i) + (k_i * exp(g_last_i - g_i))^T @ v_new_i
#
# Four matmuls per chunk, all with a contraction dimension of at most 128.
#
# Every operand is pre-arranged host-side so each matmul is a direct
# ``nc_matmul(dst, stationary, moving) == stationary.T @ moving`` with no
# in-kernel transposes: what the tensor engine wants on its partition axis is
# the contraction dimension.


_gdn_chunk_scan_kernel = None
_wrapped_gdn_chunk_scan = None

try:  # pragma: no cover - depends on the installed nki
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.nki_hop import wrap_nki

    @nki.jit
    def _gdn_chunk_scan_kernel(
        q_g_T,          # [rows, chunks, k_dim, chunk]  stationary: q_i * exp(g_i)
        k_cumdecay_T,   # [rows, chunks, k_dim, chunk]  stationary: k_cumdecay_i
        attn_T,         # [rows, chunks, chunk, chunk]  stationary: attn_i^T
        k_decay,        # [rows, chunks, chunk, k_dim]  stationary: k_i*exp(g_last-g_i)
        v_base,         # [rows, chunks, chunk, v_dim]  moving: attn_i @ v_beta_i
        g_last_rep,     # [rows, chunks, k_dim, 1]      exp(g_last_i), per-partition
        state_in,       # [rows, k_dim, v_dim]
    ):
        """Sequential inter-chunk scan of the gated delta rule.

        ``rows`` is batch and head folded together, so one launch covers a whole
        layer -- the alternative, a Python loop over 48 heads, would recreate
        exactly the per-call-site fan-out that made a three-layer DeepSeek-V4
        graph take 2h52m to compile without emitting a NEFF.

        Two nested ``nl.fori_loop``s, so the body is emitted once regardless of
        head count or sequence length. ``nl.affine_range`` and
        ``nl.static_range`` both unroll; only ``fori_loop`` does not.

        Loop indices arrive as VirtualRegisters. ``int * reg`` and ``reg + reg``
        are both rejected by the tracer, so index arithmetic is done in the SBUF
        domain -- spill with ``register_store``, combine with
        ``tensor_scalar``/``tensor_tensor``, reload with ``register_load`` --
        and every access is register-offset ``.ap()``. Indexing ``tensor[i]``
        with a register cannot narrow a leading dimension, which is the
        unresolved blocker still sitting in ``nki_mla.py``.
        """
        rows, chunks, k_dim, chunk = q_g_T.shape
        v_dim = v_base.shape[-1]

        # Tuples, never frozensets: the tracer rejects `in <frozenset>`.
        assert k_dim in (16, 32, 64, 128)
        assert v_dim in (16, 32, 64, 128)
        assert chunk in (16, 32, 64, 128)

        # Split (batch, head) rows across the LNC programs. Rows are fully
        # independent -- each carries its own recurrent state and never reads
        # another's -- so this needs no cross-program communication.
        #
        # The kernel is traced once per program with a distinct program_id
        # ("kernel is traced LNC times with different program_id_value",
        # nki/_backends/mlir_tracer), so program_id and the derived bounds are
        # trace-time Python ints, not registers. Folding the base into the loop
        # bounds is what keeps it that way: `row_start + r` inside the body
        # would be `int + VirtualRegister`, which the tracer rejects.
        n_programs = nl.num_programs(0)
        program_id = nl.program_id(0)
        assert rows % n_programs == 0
        rows_per_program = rows // n_programs
        row_start = program_id * rows_per_program

        out = nl.ndarray(
            (rows, chunks, chunk, v_dim), dtype=nl.float32, buffer=nl.shared_hbm
        )
        state_out = nl.ndarray(
            (rows, k_dim, v_dim), dtype=nl.float32, buffer=nl.shared_hbm
        )

        flat_q = q_g_T.reshape((rows * chunks * k_dim, chunk))
        flat_kc = k_cumdecay_T.reshape((rows * chunks * k_dim, chunk))
        flat_attn = attn_T.reshape((rows * chunks * chunk, chunk))
        flat_kd = k_decay.reshape((rows * chunks * chunk, k_dim))
        flat_v = v_base.reshape((rows * chunks * chunk, v_dim))
        flat_g = g_last_rep.reshape((rows * chunks * k_dim, 1))
        flat_out = out.reshape((rows * chunks * chunk, v_dim))
        flat_state_in = state_in.reshape((rows * k_dim, v_dim))
        flat_state_out = state_out.reshape((rows * k_dim, v_dim))

        def per_row(r):
            r_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.register_store(r_sb, r)

            # Row bases for this (batch, head).
            k_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_base, data=r_sb, op0=nl.multiply, operand0=chunks * k_dim
            )
            c_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=c_base, data=r_sb, op0=nl.multiply, operand0=chunks * chunk
            )
            s_base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=s_base, data=r_sb, op0=nl.multiply, operand0=k_dim
            )
            s_reg = nisa.register_alloc()
            nisa.register_load(s_reg, s_base)

            state = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=state,
                src=flat_state_in.ap(
                    pattern=[[v_dim, k_dim], [1, v_dim]],
                    scalar_offset=s_reg,
                    indirect_dim=0,
                ),
            )

            def scan_chunk(i):
                i_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.register_store(i_sb, i)

                k_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=k_off, data=i_sb, op0=nl.multiply, operand0=k_dim
                )
                nisa.tensor_tensor(dst=k_off, data1=k_off, data2=k_base, op=nl.add)
                k_reg = nisa.register_alloc()
                nisa.register_load(k_reg, k_off)

                c_off = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=c_off, data=i_sb, op0=nl.multiply, operand0=chunk
                )
                nisa.tensor_tensor(dst=c_off, data1=c_off, data2=c_base, op=nl.add)
                c_reg = nisa.register_alloc()
                nisa.register_load(c_reg, c_off)

                q_t = nl.ndarray((k_dim, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=q_t,
                    src=flat_q.ap(
                        pattern=[[chunk, k_dim], [1, chunk]],
                        scalar_offset=k_reg,
                        indirect_dim=0,
                    ),
                )
                kc_t = nl.ndarray((k_dim, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=kc_t,
                    src=flat_kc.ap(
                        pattern=[[chunk, k_dim], [1, chunk]],
                        scalar_offset=k_reg,
                        indirect_dim=0,
                    ),
                )
                attn_t = nl.ndarray((chunk, chunk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=attn_t,
                    src=flat_attn.ap(
                        pattern=[[chunk, chunk], [1, chunk]],
                        scalar_offset=c_reg,
                        indirect_dim=0,
                    ),
                )
                kd = nl.ndarray((chunk, k_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=kd,
                    src=flat_kd.ap(
                        pattern=[[k_dim, chunk], [1, k_dim]],
                        scalar_offset=c_reg,
                        indirect_dim=0,
                    ),
                )
                v_sb = nl.ndarray((chunk, v_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=v_sb,
                    src=flat_v.ap(
                        pattern=[[v_dim, chunk], [1, v_dim]],
                        scalar_offset=c_reg,
                        indirect_dim=0,
                    ),
                )
                g_last = nl.ndarray((k_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=g_last,
                    src=flat_g.ap(
                        pattern=[[1, k_dim], [1, 1]],
                        scalar_offset=k_reg,
                        indirect_dim=0,
                    ),
                )

                # v_new = v_i - k_cumdecay_i @ S
                psum_vp = nl.ndarray((chunk, v_dim), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(psum_vp, kc_t, state)
                v_new = nl.ndarray((chunk, v_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=v_new, data1=v_sb, data2=psum_vp, op=nl.subtract
                )

                # out_i = (q_i * exp(g_i)) @ S + attn_i @ v_new
                psum_inter = nl.ndarray(
                    (chunk, v_dim), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(psum_inter, q_t, state)
                inter = nl.ndarray((chunk, v_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=inter, src=psum_inter)

                psum_intra = nl.ndarray(
                    (chunk, v_dim), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(psum_intra, attn_t, v_new)

                chunk_out = nl.ndarray(
                    (chunk, v_dim), dtype=nl.float32, buffer=nl.sbuf
                )
                # At most one PSUM operand per tensor_tensor.
                nisa.tensor_tensor(
                    dst=chunk_out, data1=inter, data2=psum_intra, op=nl.add
                )
                nisa.dma_copy(
                    dst=flat_out.ap(
                        pattern=[[v_dim, chunk], [1, v_dim]],
                        scalar_offset=c_reg,
                        indirect_dim=0,
                    ),
                    src=chunk_out,
                )

                # S = S * exp(g_last) + k_decay_i^T @ v_new
                psum_state = nl.ndarray(
                    (k_dim, v_dim), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(psum_state, kd, v_new)
                decayed = nl.ndarray((k_dim, v_dim), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=decayed, data=state, op0=nl.multiply, operand0=g_last
                )
                nisa.tensor_tensor(
                    dst=state, data1=decayed, data2=psum_state, op=nl.add
                )

            nl.fori_loop(0, chunks, scan_chunk)
            nisa.dma_copy(
                dst=flat_state_out.ap(
                    pattern=[[v_dim, k_dim], [1, v_dim]],
                    scalar_offset=s_reg,
                    indirect_dim=0,
                ),
                src=state,
            )

        nl.fori_loop(row_start, row_start + rows_per_program, per_row)
        return out, state_out

    _wrapped_gdn_chunk_scan = wrap_nki(_gdn_chunk_scan_kernel)
except Exception as exc:  # noqa: BLE001
    logger.debug("GDN chunk-scan kernel unavailable: %s", exc)


# ---------------------------------------------------------------------------
# Host-side preparation and dispatch
# ---------------------------------------------------------------------------


def _prepare_chunk_scan(query, key, value, g, beta, chunk_size, use_qk_l2norm):
    """Everything the chunked delta rule does *before* the sequential loop.

    All of it is independent per chunk, so it stays in torch: batched tensor ops
    trace to a fixed graph regardless of sequence length. Only the recurrence
    needs a kernel.

    Operands come back pre-transposed into the layout ``nc_matmul`` wants -- the
    contraction dimension on the partition axis -- so the kernel performs no
    transposes of its own.

    ``l2norm`` and ``unit_triangular_inverse`` are imported here rather than at
    module scope because ``gated_deltanet`` imports this module for the conv;
    sharing them is deliberate, and it earned its keep: the inverse turned out to
    be numerically unusable on near-identical keys, and because both paths import
    the one function, the device path was fixed by fixing the oracle.

    Note the pin against HuggingFace's forward-substitution loop did *not* catch
    that -- it compares on random matrices, where the failure cannot appear. The
    test that pins it is the near-identical-keys regression in the oracle suite.

    What the two paths must *not* share is the scan itself, which is what the
    simulator test diffs.
    """
    from .gated_deltanet import l2norm, unit_triangular_inverse

    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    )
    batch, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]

    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    query = query * (k_dim**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    def to_chunks(x):
        return x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])

    q_c, k_c, k_beta_c, v_beta_c = (to_chunks(x) for x in (query, key, k_beta, v_beta))
    g_c = g.reshape(batch, heads, -1, chunk_size).cumsum(dim=-1)

    decay_mask = (g_c.unsqueeze(-1) - g_c.unsqueeze(-2)).tril().exp().float().tril()
    strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    a = -((k_beta_c @ k_c.transpose(-1, -2)) * decay_mask).masked_fill(strict_upper, 0)
    attn = unit_triangular_inverse(a)

    v_base = attn @ v_beta_c
    k_cumdecay = attn @ (k_beta_c * g_c.exp().unsqueeze(-1))

    causal = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )
    attn_i = (q_c @ k_c.transpose(-1, -2) * decay_mask).masked_fill(causal, 0)

    g_last = g_c[..., -1, None]
    q_g = q_c * g_c.unsqueeze(-1).exp()
    k_decay = k_c * (g_last.unsqueeze(-1) - g_c.unsqueeze(-1)).exp()

    rows = batch * heads
    num_chunks = q_c.shape[2]

    def fold(x):
        return x.reshape(rows, *x.shape[2:]).contiguous()

    return {
        "q_g_T": fold(q_g.transpose(-1, -2)),
        "k_cumdecay_T": fold(k_cumdecay.transpose(-1, -2)),
        "attn_T": fold(attn_i.transpose(-1, -2)),
        "k_decay": fold(k_decay),
        "v_base": fold(v_base),
        "g_last_rep": fold(
            g_last.exp()
            .reshape(batch, heads, num_chunks, 1, 1)
            .expand(batch, heads, num_chunks, k_dim, 1)
        ),
        "batch": batch,
        "heads": heads,
        "num_chunks": num_chunks,
        "seq_len": seq_len,
        "k_dim": k_dim,
        "v_dim": v_dim,
    }


#: LNC grid for the scan. Fixed at 2, never data-dependent.
#:
#: A grid=1 launch does not compile on an LNC=2 host. Under LNC=2 a logical
#: NeuronCore is two physical cores sharing an address space, and codegen is
#: emitted and checked per physical core, so a single program puts the kernel
#: body on core 0 and leaves core 1 a stub -- which neuronx-cc rejects:
#:
#:     [NCC_IXGM002] Expected function sg0000 in subgraph 0 to have 49 basic
#:     blocks, but on core 1 it has 1 basic blocks
#:
#: Isolated on device at fixed TP=4 by varying only the grid: 2 generates
#: tokens, 1 reproduces that error. This used to be ``2 if rows % 2 == 0 else
#: 1``, which made the grid a function of the per-rank head count and so made
#: the kernel silently uncompilable at TP=8 (3 value heads per rank). Rows are
#: padded to an even count instead -- see :func:`pad_rows_for_lnc`. Mirrors the
#: conv's ``_LNC``, which was never affected because it is already a constant.
_SCAN_LNC = 2


def pad_rows_for_lnc(
    tensors: list[torch.Tensor], rows: int
) -> tuple[list[torch.Tensor], int]:
    """Pad row-leading scan inputs so the row count divides ``_SCAN_LNC``.

    The scan folds ``(batch, head)`` into independent rows -- each carries its
    own recurrent state and never reads another's, which is why the split needs
    no cross-program communication -- so an appended zero row is inert. It
    computes its own zero output and is sliced away afterwards.

    That is what lets the grid stay fixed for every geometry, including the odd
    per-rank head counts the real model reaches (3 at TP=8, 1 at TP=32) which
    would otherwise drop to a grid of 1 and fail to compile.

    Mirrors ``deepseek_v4/nki_compressor.py``, which pads one inert candidate
    for odd shapes so both LNC2 programs get the same runtime-loop bound.

    Returns the (possibly unchanged) tensors and the padded row count.
    """
    pad = (-rows) % _SCAN_LNC
    if not pad:
        return list(tensors), rows
    return (
        [torch.cat((t, torch.zeros_like(t[:pad])), dim=0) for t in tensors],
        rows + pad,
    )


def can_use_chunk_scan_kernel(query: torch.Tensor, chunk_size: int) -> bool:
    """Whether the scan kernel can serve this call."""
    if _wrapped_gdn_chunk_scan is None:
        return False
    if not can_run_kernel(query):
        return False
    # Partition-axis bound, and the tracer's assert list.
    return chunk_size in (16, 32, 64, 128)


def chunk_gated_delta_rule_nki(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Device path for the chunked gated delta rule.

    Same contract as ``gated_deltanet.chunk_gated_delta_rule``: inputs
    ``[B, T, H, D]``, returns ``(output, final_state)``.
    """
    initial_dtype = query.dtype
    prep = _prepare_chunk_scan(query, key, value, g, beta, chunk_size, use_qk_l2norm)

    rows = prep["batch"] * prep["heads"]
    k_dim, v_dim = prep["k_dim"], prep["v_dim"]

    if initial_state is None:
        state_in = torch.zeros(
            rows, k_dim, v_dim, dtype=torch.float32, device=query.device
        )
    else:
        state_in = (
            initial_state.reshape(rows, k_dim, v_dim).to(torch.float32).contiguous()
        )

    kernel_inputs, _ = pad_rows_for_lnc(
        [
            prep["q_g_T"],
            prep["k_cumdecay_T"],
            prep["attn_T"],
            prep["k_decay"],
            prep["v_base"],
            prep["g_last_rep"],
            state_in,
        ],
        rows,
    )

    out, state_out = _wrapped_gdn_chunk_scan[_SCAN_LNC](*kernel_inputs)

    # Drop the inert padding row, if any, before restoring (batch, head).
    # rows is a trace-time Python int, so this stays a static slice.
    out = out[:rows]
    state_out = state_out[:rows]

    batch, heads, seq_len = prep["batch"], prep["heads"], prep["seq_len"]
    out = out.reshape(batch, heads, -1, v_dim)[:, :, :seq_len]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state_out.reshape(batch, heads, k_dim, v_dim)
