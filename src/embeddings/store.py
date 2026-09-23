"""On-disk storage of catalog embeddings, metadata and the heads used to make them.

Layout of ``artifacts/embeddings/``::

    image_raw.npy, text_raw.npy      backbone embeddings (reused by training)
    image.npy, text.npy              projected + normalised (projection_dim)
    fused.npy                        fused multimodal embeddings
    masks.npz                        has_image / has_text per row
    metadata.csv                     catalog rows; row i <-> embedding row i
    projection_heads.pt              the exact heads used for image.npy/text.npy
    manifest.json                    model names, dims, fusion settings
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings.fusion import fuse
from src.utils.config import Config

METADATA_COLUMNS = ["product_id", "title", "clean_title", "image_path", "image_file",
                    "category", "label_group", "has_image", "has_text"]


class EmbeddingStoreError(RuntimeError):
    """Raised when stored embeddings are missing or inconsistent."""


@dataclass
class EmbeddingStore:
    """All embeddings for a catalog, aligned row-by-row with ``metadata``."""

    metadata: pd.DataFrame
    image_raw: np.ndarray
    text_raw: np.ndarray
    image: np.ndarray
    text: np.ndarray
    fused: np.ndarray
    image_mask: np.ndarray
    text_mask: np.ndarray
    manifest: dict

    def __len__(self) -> int:
        return len(self.metadata)

    def refuse(self, method: str, image_weight: float, text_weight: float) -> np.ndarray:
        """Recompute fused embeddings with different fusion settings (for ablations)."""
        return fuse(self.image, self.text, method, image_weight, text_weight,
                    self.image_mask, self.text_mask)

    def save(self, directory: str | Path) -> None:
        """Write all arrays, metadata and manifest to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name in ("image_raw", "text_raw", "image", "text", "fused"):
            np.save(d / f"{name}.npy", getattr(self, name))
        np.savez(d / "masks.npz", image=self.image_mask, text=self.text_mask)
        cols = [c for c in METADATA_COLUMNS if c in self.metadata.columns]
        self.metadata[cols].to_csv(d / "metadata.csv", index=False)
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> EmbeddingStore:
        """Load a store written by :meth:`save`, validating row alignment.

        Raises:
            EmbeddingStoreError: If files are missing or row counts disagree.
        """
        d = Path(directory)
        required = ["image_raw.npy", "text_raw.npy", "image.npy", "text.npy", "fused.npy",
                    "masks.npz", "metadata.csv", "manifest.json"]
        missing = [f for f in required if not (d / f).is_file()]
        if missing:
            raise EmbeddingStoreError(
                f"embeddings in {d} are incomplete (missing {missing}); "
                "run `python scripts/build_embeddings.py` first")
        arrays = {n: np.load(d / f"{n}.npy") for n in ("image_raw", "text_raw", "image", "text", "fused")}
        masks = np.load(d / "masks.npz")
        metadata = pd.read_csv(d / "metadata.csv", dtype={"product_id": str}, keep_default_na=False)
        n = len(metadata)
        bad = {k: v.shape[0] for k, v in arrays.items() if v.shape[0] != n}
        if bad or masks["image"].shape[0] != n or masks["text"].shape[0] != n:
            raise EmbeddingStoreError(f"row count mismatch with metadata ({n} rows): {bad}")
        return cls(metadata=metadata, image_mask=masks["image"].astype(bool),
                   text_mask=masks["text"].astype(bool),
                   manifest=json.loads((d / "manifest.json").read_text()), **arrays)


def embeddings_dir(config: Config) -> Path:
    """Absolute embeddings directory from the config."""
    return config.resolve(config.paths.embeddings_dir)
