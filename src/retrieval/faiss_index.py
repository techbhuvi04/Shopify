"""FAISS index wrapper with validation and persistence.

Why ``IndexFlatIP``? All stored vectors are L2-normalised, so inner product
equals cosine similarity. A flat index performs exact search, which is fast
enough for catalogs up to a few hundred thousand items on CPU, has no training
step and no recall loss. For larger catalogs it can be swapped for an IVF/HNSW
index behind the same interface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

INDEX_NAMES = ("image", "text", "fused")


class InvalidIndexError(RuntimeError):
    """Raised when a FAISS index is missing, corrupt or inconsistent."""


def _as_float32_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {x.shape}")
    return np.ascontiguousarray(x)


class FaissIndex:
    """Thin wrapper around a FAISS inner-product index.

    FAISS row ``i`` corresponds to catalog/metadata row ``i``.
    """

    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @property
    def dim(self) -> int:
        return int(self.index.d)

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @classmethod
    def build(cls, embeddings: np.ndarray, index_type: str = "inner_product") -> FaissIndex:
        """Build an index over ``embeddings`` (``(n, d)``, expected L2-normalised).

        Raises:
            ValueError: For empty input, non-finite values or unknown index types.
        """
        x = _as_float32_2d(embeddings)
        if x.shape[0] == 0:
            raise ValueError("cannot build an index from zero vectors")
        if not np.isfinite(x).all():
            raise ValueError("embeddings contain NaN or inf values")
        if index_type != "inner_product":
            raise ValueError(f"unsupported index_type: {index_type}")
        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        return cls(index)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` ``(scores, ids)`` for each query row.

        ``k`` is clipped to the index size. Scores are cosine similarities for
        normalised inputs, sorted in descending order.

        Raises:
            ValueError: If ``k`` is not positive or the query dimension is wrong.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        q = _as_float32_2d(queries)
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        k = min(k, self.ntotal)
        scores, ids = self.index.search(q, k)
        return scores, ids

    def save(self, path: str | Path) -> None:
        """Serialise the index to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, expected_dim: int | None = None,
             expected_size: int | None = None) -> FaissIndex:
        """Load and validate an index.

        Raises:
            InvalidIndexError: If the file is missing, unreadable, or its
                dimension / size does not match expectations.
        """
        path = Path(path)
        if not path.is_file():
            raise InvalidIndexError(f"FAISS index not found: {path}; run `python scripts/build_index.py`")
        try:
            index = faiss.read_index(str(path))
        except RuntimeError as exc:
            raise InvalidIndexError(f"could not read FAISS index {path}: {exc}") from exc
        wrapper = cls(index)
        if expected_dim is not None and wrapper.dim != expected_dim:
            raise InvalidIndexError(f"{path.name}: dim {wrapper.dim} != expected {expected_dim}")
        if expected_size is not None and wrapper.ntotal != expected_size:
            raise InvalidIndexError(f"{path.name}: {wrapper.ntotal} vectors != expected {expected_size}; "
                                    "the index is stale, rebuild it")
        return wrapper


def product_ids_fingerprint(product_ids: list[str]) -> str:
    """Stable hash of the ordered product IDs, used to detect stale indexes."""
    return hashlib.sha256("\n".join(map(str, product_ids)).encode()).hexdigest()[:16]


def save_index_set(directory: str | Path, indexes: dict[str, FaissIndex],
                   product_ids: list[str], extra: dict | None = None) -> None:
    """Save several named indexes plus a manifest describing them."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, index in indexes.items():
        index.save(d / f"{name}.faiss")
    manifest = {
        "n_products": len(product_ids),
        "product_ids_fingerprint": product_ids_fingerprint(product_ids),
        "indexes": {name: {"dim": idx.dim, "ntotal": idx.ntotal, "type": "IndexFlatIP"}
                    for name, idx in indexes.items()},
        **(extra or {}),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_index_set(directory: str | Path, product_ids: list[str]) -> dict[str, FaissIndex]:
    """Load all indexes in ``directory`` and verify they match ``product_ids``.

    Raises:
        InvalidIndexError: If the manifest is missing or the indexes were built
            from a different catalog than ``product_ids``.
    """
    d = Path(directory)
    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        raise InvalidIndexError(f"index manifest not found in {d}; run `python scripts/build_index.py`")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise InvalidIndexError(f"corrupt index manifest {manifest_path}: {exc}") from exc
    if manifest.get("product_ids_fingerprint") != product_ids_fingerprint(product_ids):
        raise InvalidIndexError("indexes were built from different embeddings; "
                                "rerun `python scripts/build_index.py`")
    return {
        name: FaissIndex.load(d / f"{name}.faiss", info["dim"], len(product_ids))
        for name, info in manifest["indexes"].items()
    }
