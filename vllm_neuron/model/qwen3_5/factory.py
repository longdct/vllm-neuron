# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5-family text decoder (Qwen3.5 / 3.6 / 3.8).

Validates up front so unsupported configurations fail with a clear message
rather than deep inside weight loading or, worse, silently. Follows the shape of
``qwen3/factory.py`` and ``deepseek_v4/factory.py``: subclass ``nn.Module`` to
satisfy vLLM's ``ModelRegistry``, but hand back the real implementation from the
``from_configs`` classmethod.
"""

import logging
from typing import TYPE_CHECKING

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

if TYPE_CHECKING:
    from vllm_neuron.model.neuron_config import VisionNeuronConfig

from .config import Qwen3_5TextConfig
from .parallel import resolve_sharding
from .quantization import read_quantization_spec

logger = logging.getLogger(__name__)


class Qwen3_5ForCausalLM(nn.Module):
    """Factory that validates config and selects the Qwen3.5 implementation.

    The second parameter is named ``text_neuron_config`` rather than
    ``neuron_config`` to match what the runner actually calls. Every released
    Qwen3.5-family checkpoint carries a ``vision_config``, which makes
    ``platform.py::_resolve_vision_auto_config`` synthesize a
    ``vision_neuron_config``; ``neuron_model_runner.load_model`` then takes its
    multimodal branch and calls
    ``from_configs(hf_config=..., text_neuron_config=..., vision_neuron_config=...)``
    by keyword. A ``(hf_config, neuron_config)`` signature raises
    ``TypeError: got an unexpected keyword argument 'text_neuron_config'`` on
    every rank, at every TP degree including 1 -- so no released checkpoint
    could load at all. The positional two-argument call used by the text-only
    branch still works unchanged.

    Mirrors ``qwen3_vl/factory.py``, which already carries this signature.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: "VisionNeuronConfig | None" = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: "VisionNeuronConfig | None" = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
        vision_neuron_config: "VisionNeuronConfig | None" = None,
    ) -> nn.Module:
        config = Qwen3_5TextConfig.from_configs(hf_config, neuron_config)
        # ``quantization_config`` lives on the multimodal wrapper config and
        # never reaches ``text_config``, so it is read here and attached
        # rather than in ``from_configs``.
        config.quant_spec = read_quantization_spec(hf_config)
        cls._validate_config(config, neuron_config)

        if vision_neuron_config is not None:
            # Accepted so the checkpoint loads, then dropped: this module is the
            # text decoder only, and no vision tower is built. Multimodal input
            # is therefore not served -- the weights under ``model.visual.*``
            # are skipped by prefix at load time. Warn rather than raise,
            # because the runner supplies this config from the mere presence of
            # hf_config.vision_config, not because the caller asked for images.
            logger.warning(
                "Qwen3.5: a vision_neuron_config was supplied but the vision "
                "tower is not implemented; serving the text decoder only. "
                "Image and video inputs will not work."
            )

        from .model import Qwen3_5TextForCausalLM as Model

        return Model.from_configs(config, neuron_config)

    @classmethod
    def _validate_config(
        cls, config: Qwen3_5TextConfig, neuron_config: NeuronConfig | None
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16", "fp8"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for the "
                "Qwen3.5 family. Implemented: None/'bf16', and 'fp8' "
                "(per-channel weights, runtime activation scales)."
            )

        if quantization == "fp8":
            # Say the cost out loud. Quantizing to e4m3 is lossy in a way the
            # BF16 path is not: measured on the real 27B, each quantized
            # weight tensor carries ~2.6% relative error. That is normal for
            # weight-only FP8 and usually harmless, but it is a change in
            # output that no startup check can verify.
            logger.warning(
                "Qwen3.5: serving with per-channel FP8 weights. The MLP and "
                "full-attention projections are quantized (~70%% of weights); "
                "GatedDeltaNet layers, embeddings and lm_head stay BF16. "
                "Expect ~2.6%% relative error per quantized tensor -- validate "
                "output quality against the BF16 baseline before deploying."
            )

        # A quantized checkpoint must be refused, not tolerated. Nothing
        # downstream reads scale tensors: ``text_weight_mappings`` maps only
        # ``.weight`` keys, so ``.weight_scale_inv`` is dropped as an
        # unexpected key, and the loader then casts the raw fp8 bytes to
        # bfloat16. That produces fluent, confidently wrong output with no
        # error -- the worst possible failure. Raise instead.
        spec = config.quant_spec
        if spec is not None and spec.is_quantized():
            raise ValueError(
                f"checkpoint declares {spec.weight_format.value} weights, but "
                "the Qwen3.5 BF16 implementation cannot read them: the "
                "per-block scale tensors would be discarded and the FP8 bytes "
                "reinterpreted as BF16, which produces plausible-looking "
                "garbage rather than an error. Serve the unquantized BF16 "
                "checkpoint instead."
            )

        # Multi-token prediction is out of scope. This used to raise, on the
        # premise that no released checkpoint ships MTP weights so honouring
        # the config would build a subtree nothing could fill. That premise is
        # wrong: Qwen3.5-0.8B and Qwen3.8-27B both ship 15 `mtp.*` tensors, and
        # both declare mtp_num_hidden_layers=1. Raising therefore made every
        # real checkpoint unservable at every TP degree.
        #
        # Warn instead. Nothing here builds an MTP subtree -- this field is read
        # nowhere else in the model -- so the `mtp.*` weights are simply skipped
        # by prefix at load time, exactly as `model.visual.*` is, and the
        # decoder serves normally without speculative decoding.
        if config.mtp_num_hidden_layers:
            logger.warning(
                "Qwen3.5: config declares mtp_num_hidden_layers=%d, but "
                "multi-token prediction is not implemented. The mtp.* weights "
                "are skipped and the model serves as a plain decoder; "
                "speculative decoding is unavailable.",
                config.mtp_num_hidden_layers,
            )

        # Resolve the sharding policy now so an unsupported TP degree is a
        # startup error rather than a silently wrong shard.
        tp_degree = cls._tp_degree()
        resolve_sharding(config, tp_degree)

        if config.needs_single_shot_prefill:
            # head_dim > 128 cannot use the segmented-attention kernel at all
            # (it raises), so chunked prefill is unavailable for the
            # full-attention layers.
            #
            # This has to *raise*, not warn. Unlike `qwen3`, whose attention
            # dispatches on `kv_segment_size` and calls NF.segmented_attention
            # (qwen3/model.py:321,363), `Qwen3_5Attention.forward_prefill` has
            # no segmented branch at all: it runs one flash_attention over the
            # tokens it was handed. Handed a chunk, it attends within that chunk
            # and silently ignores everything cached before it -- coherent,
            # confident, wrong output rather than a failure. A warning is not
            # enough protection against that.
            #
            # `kv_segment_size_buckets` being set is the signal that segmented
            # prefill is on; the engine fills it in from
            # `resolve_segmented_prefill_config` whenever
            # max_num_batched_tokens < max_model_len.
            if neuron_config is not None and neuron_config.kv_segment_size_buckets:
                raise ValueError(
                    f"Qwen3.5 head_dim={config.head_dim} exceeds the "
                    "segmented-attention kernel's 128-element partition bound, "
                    "so chunked/segmented prefill is unavailable -- but "
                    f"kv_segment_size_buckets="
                    f"{neuron_config.kv_segment_size_buckets} requests it. "
                    "Set max_num_batched_tokens == max_model_len (single-shot "
                    "prefill) and keep max_model_len within the single-shot "
                    "limit."
                )
            logger.warning(
                "Qwen3.5 head_dim=%d exceeds the segmented-attention kernel's "
                "128-element partition bound, so chunked/segmented prefill is "
                "unavailable and prefill must be single-shot. Keep "
                "max_model_len within the single-shot limit.",
                config.head_dim,
            )

    @staticmethod
    def _tp_degree() -> int:
        """Tensor-parallel degree from vLLM, defaulting to 1 outside an engine."""
        try:
            from vllm.config import get_current_vllm_config

            return get_current_vllm_config().parallel_config.tensor_parallel_size
        except Exception:  # pragma: no cover - unit tests run outside an engine
            return 1
