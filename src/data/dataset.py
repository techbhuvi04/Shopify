"""Product catalog loading, validation and leakage-safe splitting.

Why group-based splits?
    In product matching, every ``label_group`` is a set of listings of the
    *same* product. If listings of one group land in both train and test, a
    model (or a tuned threshold) is evaluated on products it has effectively
    already seen, which inflates metrics. Splitting by ``label_group`` keeps
    every product entirely on one side, mimicking the real use case of
    matching listings of *new* products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from src.data.preprocessing import clean_title
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("product_id", "title", "image_path")
OPTIONAL_COLUMNS = ("category", "label_group")


class DatasetError(ValueError):
    """Raised when a product CSV is missing or malformed."""


@dataclass
class ProductCatalog:
    """A validated product table.

    Attributes:
        df: DataFrame with the original columns plus ``clean_title``,
            ``image_file`` (absolute path), ``has_image`` and ``has_text``.
            Row order defines the embedding / FAISS index position.
    """

    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    @property
    def product_ids(self) -> list[str]:
        return self.df["product_id"].tolist()

    @property
    def has_labels(self) -> bool:
        return "label_group" in self.df.columns and self.df["label_group"].notna().all()

    @property
    def labels(self) -> np.ndarray:
        """Integer-encoded ``label_group`` per row (requires labels)."""
        if not self.has_labels:
            raise DatasetError("catalog has no complete label_group column")
        return pd.factorize(self.df["label_group"])[0]


def load_catalog(csv_path: str | Path, image_root: str | Path | None = None,
                 clean: bool = True) -> ProductCatalog:
    """Load and validate a product CSV.

    Expected columns: ``product_id, title, image_path`` (required) and
    ``category, label_group`` (optional; ``label_group`` is needed for
    evaluation and training).

    Missing image files and empty titles do not abort loading: they are
    flagged via ``has_image`` / ``has_text`` so downstream code can fall back
    to the available modality.

    Args:
        csv_path: Path to the CSV file.
        image_root: Directory that relative ``image_path`` values are relative
            to. Defaults to the CSV's directory.
        clean: Apply :func:`clean_title` to titles.

    Raises:
        DatasetError: If the file is missing, empty, lacks required columns or
            has duplicate product IDs.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetError(f"product CSV not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path, dtype={"product_id": str}, keep_default_na=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DatasetError(f"could not parse {csv_path}: {exc}") from exc
    if df.empty:
        raise DatasetError(f"product CSV is empty: {csv_path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"{csv_path} is missing required columns: {missing}")
    if df["product_id"].isna().any():
        raise DatasetError("product_id must not be empty")
    dupes = df.loc[df["product_id"].duplicated(), "product_id"].unique()
    if len(dupes):
        raise DatasetError(f"duplicate product_id values: {list(dupes[:5])}")

    root = Path(image_root) if image_root is not None else csv_path.parent
    df = df.reset_index(drop=True).copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["clean_title"] = [clean_title(t) if clean else str(t).strip() for t in df["title"]]
    df["image_file"] = [
        str((root / p).resolve()) if isinstance(p, str) and p.strip() else ""
        for p in df["image_path"]
    ]
    df["has_image"] = [bool(p) and Path(p).is_file() for p in df["image_file"]]
    df["has_text"] = df["clean_title"].str.len() > 0

    n_no_img, n_no_txt = int((~df["has_image"]).sum()), int((~df["has_text"]).sum())
    if n_no_img:
        logger.warning("%d/%d products have a missing or unreadable image path", n_no_img, len(df))
    if n_no_txt:
        logger.warning("%d/%d products have an empty title", n_no_txt, len(df))
    both_missing = ~(df["has_image"] | df["has_text"])
    if both_missing.any():
        logger.warning("%d products have neither image nor title; they cannot be matched",
                       int(both_missing.sum()))
    logger.info("loaded %d products from %s", len(df), csv_path)
    return ProductCatalog(df=df)


def load_catalog_from_config(config: Config) -> ProductCatalog:
    """Load the catalog described by ``config.data``."""
    return load_catalog(config.resolve(config.data.products_csv),
                        image_root=config.resolve(config.data.image_root),
                        clean=config.data.clean_titles)


def split_indices(catalog: ProductCatalog, strategy: str = "group_shuffle",
                  test_size: float = 0.3, n_splits: int = 5, fold: int = 0,
                  seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split catalog rows into train / test row indices.

    Args:
        strategy: ``group_shuffle`` (GroupShuffleSplit on ``label_group``),
            ``group_kfold`` (fold ``fold`` of GroupKFold is the test set) or
            ``random`` (row-level; leaks groups — only for comparison).
        test_size: Test fraction for ``group_shuffle`` / ``random``.
        n_splits: Number of folds for ``group_kfold``.
        fold: Test fold index for ``group_kfold``.
        seed: Random seed.

    Returns:
        ``(train_idx, test_idx)`` as sorted integer arrays.
    """
    n = len(catalog)
    idx = np.arange(n)
    if strategy == "random":
        logger.warning("random split ignores label_group and can leak products across splits")
        train, test = train_test_split(idx, test_size=test_size, random_state=seed)
        return np.sort(train), np.sort(test)
    if not catalog.has_labels:
        raise DatasetError(f"split strategy '{strategy}' requires a complete label_group column")
    groups = catalog.df["label_group"].to_numpy()
    if strategy == "group_shuffle":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(idx, groups=groups))
    elif strategy == "group_kfold":
        n_groups = len(np.unique(groups))
        if n_splits > n_groups:
            raise DatasetError(f"n_splits={n_splits} exceeds number of groups ({n_groups})")
        folds = list(GroupKFold(n_splits=n_splits).split(idx, groups=groups))
        train, test = folds[fold]
    else:
        raise DatasetError(f"unknown split strategy: {strategy}")
    return np.sort(train), np.sort(test)


def split_from_config(catalog: ProductCatalog, config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Split using ``config.split`` and ``config.seed``."""
    s = config.split
    return split_indices(catalog, s.strategy, s.test_size, s.n_splits, s.fold, config.seed)
