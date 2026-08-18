# SPDX-License-Identifier: Apache-2.0
import os

from .llama3 import LlamaForCausalLM
from .gpt_oss import GptOssForCausalLM
from .llama3 import Eagle3LlamaForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_vl import Qwen3VLForConditionalGeneration


def get_models() -> list[tuple[str, type]]:
    """Return a list of available model classes.

    Returns:
        list[tuple[str, type]]: A list of tuples containing model names and their corresponding classes.
            Each tuple contains (model_name, model_class) where:
            - model_name (str): The string identifier for the model, compatible with Hugging Face transformers architecture
            - model_class (type): The actual model class implementation
    """
    models = [
        ("LlamaForCausalLM", LlamaForCausalLM),
        ("GptOssForCausalLM", GptOssForCausalLM),
        ("Eagle3LlamaForCausalLM", Eagle3LlamaForCausalLM),
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),
    ]

    # SyntheticNeuronModel is a testing-only model that replaces real neural
    # network computation with deterministic KV cache fill/verify. Useful for
    # validating infrastructure (KV transfer, sharding, block management)
    # without requiring model weights or compilation.
    # Not for production inference — gated to avoid exposing to customers.
    if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
        from .synthetic import SyntheticNeuronModel

        models.append(("SyntheticNeuronModel", SyntheticNeuronModel))

    # DeepseekV4ForCausalLM is gated the same way: registering it
    # unconditionally would advertise production support that doesn't exist
    # yet. Steps 1-3 of docs/model-dev/deepseek-v4-serving-roadmap.md are
    # done (device-shaped forward, real paged cache I/O including the
    # compressor carry-cache, TP/EP scaffolding), validated against the
    # tiny/synthetic config -- but real checkpoint loading, quantization, and
    # memory calibration (roadmap steps 4-5) are not. This lets our own
    # end-to-end tests exercise the real vllm.LLM() serving path without
    # exposing the model to default users.
    if os.environ.get("VLLM_NEURON_ENABLE_DEEPSEEK_V4") == "1":
        from .deepseek_v4.factory import DeepseekV4ForCausalLM

        models.append(("DeepseekV4ForCausalLM", DeepseekV4ForCausalLM))

    return models
