"""Typed configuration loading.

The YAML file in ``configs/config.yaml`` is the single source of truth for
model names, fusion weights, thresholds and paths. This module parses it into
dataclasses so the rest of the code base gets attribute access, type hints and
validation instead of passing raw dictionaries around.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"
CONFIG_ENV_VAR = "MPM_CONFIG"
DEVICE_ENV_VAR = "MPM_DEVICE"

VALID_MODES = ("image", "text", "multimodal")


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass
class ModelConfig:
    image_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    projection_dim: int = 256
    projection_init: str = "orthogonal"
    projection_weights: str | None = None
    image_batch_size: int = 32
    text_batch_size: int = 64


@dataclass
class FusionConfig:
    method: str = "weighted_sum"
    image_weight: float = 0.6
    text_weight: float = 0.4

    @property
    def normalized_weights(self) -> tuple[float, float]:
        """Return ``(image_weight, text_weight)`` rescaled to sum to one."""
        total = self.image_weight + self.text_weight
        return self.image_weight / total, self.text_weight / total


@dataclass
class RetrievalConfig:
    index_type: str = "inner_product"
    top_k: int = 10
    candidate_multiplier: int = 5


@dataclass
class MatchingConfig:
    threshold: float = 0.75
    mode_thresholds: dict[str, float | None] = field(default_factory=dict)

    def threshold_for(self, mode: str) -> float:
        """Return the decision threshold for a retrieval mode."""
        value = self.mode_thresholds.get(mode)
        return float(value) if value is not None else float(self.threshold)


@dataclass
class DataConfig:
    products_csv: str = "data/sample_products.csv"
    image_root: str = "data"
    clean_titles: bool = True


@dataclass
class SplitConfig:
    strategy: str = "group_shuffle"
    test_size: float = 0.3
    n_splits: int = 5
    fold: int = 0


@dataclass
class EvaluationConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    candidate_k: int = 50
    thresholds: list[float] = field(
        default_factory=lambda: [round(0.5 + 0.05 * i, 2) for i in range(10)]
    )


@dataclass
class TrainingConfig:
    loss: str = "triplet"
    margin: float = 0.2
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    groups_per_batch: int = 16
    samples_per_group: int = 2
    output_weights: str = "artifacts/models/projection_heads_finetuned.pt"


@dataclass
class PathsConfig:
    embeddings_dir: str = "artifacts/embeddings"
    indexes_dir: str = "artifacts/indexes"
    plots_dir: str = "artifacts/plots"
    results_dir: str = "artifacts/results"
    models_dir: str = "artifacts/models"


@dataclass
class Config:
    """Root configuration object."""

    seed: int = 42
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    root: Path = PROJECT_ROOT

    def resolve(self, path: str | Path) -> Path:
        """Resolve a (possibly relative) path against the project root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    def validate(self) -> None:
        """Check value ranges and enumerations; raise :class:`ConfigError`."""
        if self.model.projection_dim <= 0:
            raise ConfigError("model.projection_dim must be positive")
        if self.model.projection_init not in {"orthogonal", "random"}:
            raise ConfigError("model.projection_init must be 'orthogonal' or 'random'")
        if self.fusion.method not in {"weighted_sum", "concat"}:
            raise ConfigError("fusion.method must be 'weighted_sum' or 'concat'")
        if self.fusion.image_weight < 0 or self.fusion.text_weight < 0:
            raise ConfigError("fusion weights must be non-negative")
        if self.fusion.image_weight + self.fusion.text_weight <= 0:
            raise ConfigError("fusion weights must not both be zero")
        if self.retrieval.index_type != "inner_product":
            raise ConfigError("retrieval.index_type currently supports only 'inner_product'")
        if self.retrieval.top_k <= 0 or self.retrieval.candidate_multiplier <= 0:
            raise ConfigError("retrieval.top_k and candidate_multiplier must be positive")
        thresholds = [self.matching.threshold, *self.evaluation.thresholds]
        thresholds += [v for v in self.matching.mode_thresholds.values() if v is not None]
        if any(not -1.0 <= float(t) <= 1.0 for t in thresholds):
            raise ConfigError("thresholds must lie in [-1, 1] (cosine similarity range)")
        unknown_modes = set(self.matching.mode_thresholds) - set(VALID_MODES)
        if unknown_modes:
            raise ConfigError(f"unknown modes in matching.mode_thresholds: {unknown_modes}")
        if self.split.strategy not in {"group_shuffle", "group_kfold", "random"}:
            raise ConfigError("split.strategy must be group_shuffle, group_kfold or random")
        if not 0.0 < self.split.test_size < 1.0:
            raise ConfigError("split.test_size must be in (0, 1)")
        if not 0 <= self.split.fold < self.split.n_splits:
            raise ConfigError("split.fold must be in [0, n_splits)")
        if self.training.loss not in {"triplet", "contrastive"}:
            raise ConfigError("training.loss must be 'triplet' or 'contrastive'")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("device must be auto, cpu, cuda or mps")


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    """Recursively instantiate a dataclass from a dictionary, rejecting unknown keys."""
    values = values or {}
    if not isinstance(values, dict):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(values).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        kwargs[name] = _build(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Resolution order for the file: explicit ``path`` argument, then the
    ``MPM_CONFIG`` environment variable, then ``configs/config.yaml``.
    The ``MPM_DEVICE`` environment variable overrides ``device``.

    Args:
        path: Optional path to a YAML config file.

    Returns:
        A validated :class:`Config`.

    Raises:
        ConfigError: If the file is missing, unparsable or invalid.
    """
    config_path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc
    raw.pop("root", None)
    config: Config = _build(Config, raw)
    if env_device := os.environ.get(DEVICE_ENV_VAR):
        config.device = env_device
    config.validate()
    return config
