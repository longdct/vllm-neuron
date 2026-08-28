# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone
from typing import Optional, Union, Literal, Dict, Any, List
from pydantic import BaseModel, Field, validator

__all__ = ["ArtifactMetadata", "ArtifactMetadataV0", "create_metadata"]


class ArtifactMetadataV0(BaseModel):
    """Schema for artifact metadata version 0.

    This schema defines the structure of compilation artifact metadata
    stored alongside cached NEFF and HLO files. It includes essential
    information needed to correctly reconstruct CompilationArtifacts.

    Attributes:
        version: Schema version (always 0 for this class)
        created_timestamp: ISO 8601 timestamp when metadata was created
        cache_key: The deterministic hash key used to identify this cache entry
        output_count: Number of original model outputs before aliasing passes
        unused_input_indices: Indices of inputs that were DCE'd by XLA during compilation

    Examples:
        >>> metadata = ArtifactMetadataV0(
        ...     created_timestamp="2024-01-01T00:00:00+00:00",
        ...     cache_key="abc123def456abc123def456abc123de",
        ...     output_count=2,
        ...     unused_input_indices=[1, 3]
        ... )
        >>> print(metadata.model_dump_json())
        {"version": 0, "created_timestamp": "2024-01-01T00:00:00+00:00", "cache_key": "abc123def456abc123def456abc123de", "output_count": 2, "unused_input_indices": [1, 3]}
    """

    version: Literal[0] = Field(default=0, description="Metadata schema version")

    created_timestamp: str = Field(
        ..., description="ISO 8601 timestamp when metadata was created"
    )

    cache_key: str = Field(
        ..., description="Deterministic hash key identifying this cache entry"
    )

    output_count: Optional[int] = Field(
        default=None,
        description="Number of original model outputs before aliasing passes",
        ge=0,
    )

    unused_input_indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of inputs that were DCE'd by XLA during compilation",
    )

    has_rng_seed_parameter: bool = Field(
        default=False,
        descuption="Whether the graph needs additional RNG seed parameter",
    )

    io_map: Optional[Dict[int, int]] = Field(
        default=None,
        description="Input-output aliasing mapping (output index -> input index)",
    )

    @validator("unused_input_indices")
    def validate_unused_input_indices(
        cls, v: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Validate that all unused input indices are non-negative integers.

        Args:
            v: List of unused input indices to validate

        Returns:
            Optional[List[int]]: Validated list of indices

        Raises:
            ValueError: If any index is negative
        """
        if v is not None:
            for idx in v:
                if not isinstance(idx, int) or idx < 0:
                    raise ValueError(
                        f"Unused input indices must be non-negative integers, got: {idx}"
                    )
        return v

    @validator("created_timestamp")
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO 8601 timestamp format.

        Args:
            v: Timestamp string to validate

        Returns:
            str: Validated timestamp string

        Raises:
            ValueError: If timestamp is not valid ISO 8601 format
        """
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 timestamp format: {v}")
        return v

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"  # Prevent additional fields
        validate_assignment = True


# Union type for all schema versions
ArtifactMetadata = Union[ArtifactMetadataV0]


def create_metadata(
    cache_key: str,
    output_count: Optional[int] = None,
    unused_input_indices: Optional[List[int]] = None,
    has_rng_seed_parameter: bool = False,
    io_map: Optional[Dict[int, int]] = None,
) -> ArtifactMetadataV0:
    """Factory function to create metadata with current timestamp.

    Args:
        cache_key: The deterministic hash key identifying this cache entry
        output_count: Optional number of original model outputs
        unused_input_indices: Optional list of input indices that were DCE'd by XLA
        has_rng_seed_parameter: Whether the graph needs an additional RNG seed parameter
        io_map: Optional input-output aliasing mapping (output index -> input index)

    Returns:
        ArtifactMetadataV0: Initialized metadata object

    Examples:
        >>> metadata = create_metadata(cache_key="abc123def456abc123def456abc123de", output_count=3)
        >>> metadata.output_count
        3
    """
    return ArtifactMetadataV0(
        created_timestamp=datetime.now(timezone.utc).isoformat(),
        cache_key=cache_key,
        output_count=output_count,
        unused_input_indices=unused_input_indices,
        has_rng_seed_parameter=has_rng_seed_parameter,
        io_map=io_map,
    )


def parse_metadata_dict(data: Dict[str, Any]) -> ArtifactMetadata:
    """Parse metadata from dictionary with version detection.

    Args:
        data: Dictionary containing metadata fields

    Returns:
        ArtifactMetadata: Parsed and validated metadata object

    Raises:
        ValueError: If version is unsupported or validation fails

    Examples:
        >>> data = {"version": 0, "created_timestamp": "2024-01-01T00:00:00+00:00"}
        >>> metadata = parse_metadata_dict(data)
        >>> isinstance(metadata, ArtifactMetadataV0)
        True
    """
    version = data.get("version", 0)

    if version == 0:
        return ArtifactMetadataV0(**data)
    else:
        raise ValueError(f"Unsupported metadata schema version: {version}")
