"""Configuration classes for microbiome ML workflows.

This module provides comprehensive configuration classes for all major
processing workflows in the microbiome ML pipeline.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

import yaml  # type: ignore


class AggregationMethod(Enum):
    """Enumeration of available aggregation methods for sample-wise
    features."""

    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN = "harmonic_mean"
    MEDIAN = "median"
    PRESENCE_ABSENCE = "presence_absence"
    TOP_K_ABUNDANT = "top_k_abundant"
    MAX_ABUNDANCE = "max_abundance"
    MIN_ABUNDANCE = "min_abundance"
    # Legacy methods for backward compatibility
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"


class WeightingStrategy(Enum):
    """Enumeration of available weighting strategies."""

    NONE = "none"
    ABUNDANCE = "abundance"
    SQRT_ABUNDANCE = "sqrt_abundance"
    # Legacy strategies for backward compatibility
    UNIFORM = "uniform"
    INVERSE_ABUNDANCE = "inverse_abundance"
    LOG_ABUNDANCE = "log_abundance"
    PRESENCE_ABSENCE = "presence_absence"


@dataclass
class AggregationConfig:
    """Configuration for feature aggregation operations."""

    method: AggregationMethod
    weighting: WeightingStrategy = WeightingStrategy.ABUNDANCE
    uncommon_feature_cutoff: int = 10
    pseudocount: float = 1e-8
    min_abundance: float = 0.001
    k: int = 10  # for top_k_abundant method

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not isinstance(self.method, AggregationMethod):
            if isinstance(self.method, str):
                self.method = AggregationMethod(self.method)
            else:
                raise ValueError(f"Invalid aggregation method: {self.method}")

        if not isinstance(self.weighting, WeightingStrategy):
            if isinstance(self.weighting, str):
                self.weighting = WeightingStrategy(self.weighting)
            else:
                raise ValueError(
                    f"Invalid weighting strategy: {self.weighting}"
                )


class Config:
    """Configuration class that loads YAML files and provides dot notation
    access."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(self.config_path, "r") as f:
            self._data = yaml.safe_load(f)

        # Convert nested dictionaries to Config objects for dot notation
        self._convert_dicts()

    def _convert_dicts(self) -> None:
        """Convert nested dictionaries to Config objects recursively."""
        for key, value in self._data.items():
            if isinstance(value, dict):
                setattr(self, key, self._dict_to_config(value))
            else:
                setattr(self, key, value)

    def _dict_to_config(self, data: Dict[str, Any]) -> "Config":
        """Convert a dictionary to a Config object."""
        config_obj = Config.__new__(Config)  # Create without calling __init__
        config_obj._data = data

        for key, value in data.items():
            if isinstance(value, dict):
                setattr(config_obj, key, self._dict_to_config(value))
            else:
                setattr(config_obj, key, value)

        return config_obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with optional default."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        if isinstance(self._data, dict):
            return self._data.copy()
        return {}

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return hasattr(self, key)

    def __repr__(self) -> str:
        return f"Config({self._data})"
