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

    # The Qwen3.5-family text decoder (Qwen3.5 / Qwen3.6 / Qwen3.8 all declare
    # Qwen3_5ForConditionalGeneration and share one architecture) is gated for
    # the same reason as DeepSeek-V4: the hybrid Gated-DeltaNet stack is
    # validated against tiny configs only. Real-checkpoint accuracy and the
    # 27B cold compile are not done, and head_dim=256 still forces single-shot
    # prefill. This lets our own tests drive the real vllm.LLM() path without
    # advertising support that does not exist yet.
    if os.environ.get("VLLM_NEURON_ENABLE_QWEN3_5") == "1":
        from .qwen3_5.factory import Qwen3_5ForCausalLM

        models.append(("Qwen3_5ForCausalLM", Qwen3_5ForCausalLM))
        # Released checkpoints are multimodal wrappers; we serve the text
        # decoder and skip the model.visual.* subtree at load time.
        models.append(("Qwen3_5ForConditionalGeneration", Qwen3_5ForCausalLM))

    return models
