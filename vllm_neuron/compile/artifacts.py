# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_neuron.compile.schema import ArtifactMetadata


@dataclass
class CompilationArtifacts:
    """Compilation artifacts from FX -> HLO -> NEFF pipeline.

    This dataclass contains the essential outputs of the Neuron compilation
    process that are needed to create an executable.

    Attributes:
        hlo_filename: Path to the serialized HLO file (graph.hlo)
        neff_filename: Path to the compiled NEFF file (graph.neff)
        metadata: Additional compilation metadata with validated schema.
    """

    hlo_filename: str
    neff_filename: str
    metadata: "ArtifactMetadata"
