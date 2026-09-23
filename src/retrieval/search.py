"""High-level product search engine used by the API, the UI and notebooks.

Retrieval flow::

    query (image and/or title)
        -> MultimodalEncoder (same projection heads as the catalog)
        -> FAISS index for the mode (image / text / fused)
        -> candidate rows -> metadata lookup
        -> per-modality + combined similarity, threshold -> MATCH / NOT_MATCH

In multimodal mode FAISS retrieves ``top_k * candidate_multiplier`` candidates
from the fused index and they are re-ranked by the explicit combined score
``w_i * sim_img + w_t * sim_txt``. (The fused-vector inner product is close to,
but not identical to, that score because of cross-modal cross terms.)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.embeddings.store import EmbeddingStore, embeddings_dir
from src.models.multimodal_encoder import MultimodalEncoder, load_projection_heads
from src.retrieval.faiss_index import FaissIndex, load_index_set
from src.retrieval.matching import (MatchResult, match_pair, modality_weights,
                                    score_candidates)
from src.utils.config import VALID_MODES, Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODE_TO_INDEX = {"image": "image", "text": "text", "multimodal": "fused"}


class QueryError(ValueError):
    """Raised for invalid queries (e.g. image mode without an image)."""


@dataclass
class SearchResult:
    """One retrieved product."""

    rank: int
    product_id: str
    title: str
    image_path: str
    image_file: str
    category: str
    image_similarity: float | None
    text_similarity: float | None
    combined_similarity: float
    is_match: bool

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("image_file")  # absolute local path; not exposed over the API
        return d


def _clean_float(x: float | np.floating | None) -> float | None:
    """Round to 4 decimals; NaN (modality unavailable) becomes ``None``."""
    return None if x is None or math.isnan(float(x)) else round(float(x), 4)


class ProductSearchEngine:
    """Search and match products against an indexed catalog.

    Args:
        config: Project configuration.
        store: Catalog embeddings and metadata.
        indexes: FAISS indexes keyed by ``image``, ``text`` and ``fused``.
        encoder: Encoder for queries; must use the same heads as ``store``.
    """

    def __init__(self, config: Config, store: EmbeddingStore,
                 indexes: dict[str, FaissIndex], encoder: MultimodalEncoder) -> None:
        self.config = config
        self.store = store
        self.indexes = indexes
        self.encoder = encoder
        self._row_by_id = {pid: i for i, pid in enumerate(store.metadata["product_id"])}

    @classmethod
    def from_artifacts(cls, config: Config, encoder: MultimodalEncoder | None = None) -> ProductSearchEngine:
        """Load embeddings, indexes and projection heads from ``artifacts/``.

        Raises:
            EmbeddingStoreError / InvalidIndexError: If artifacts are missing
                or inconsistent with each other.
        """
        emb_dir = embeddings_dir(config)
        store = EmbeddingStore.load(emb_dir)
        indexes = load_index_set(config.resolve(config.paths.indexes_dir),
                                 store.metadata["product_id"].tolist())
        encoder = encoder or MultimodalEncoder(config)
        image_head, text_head, _ = load_projection_heads(emb_dir / "projection_heads.pt")
        encoder.set_heads(image_head, text_head)
        if store.manifest.get("fusion", {}).get("method", config.fusion.method) != config.fusion.method:
            logger.warning("config fusion.method differs from the one used to build the index; "
                           "rebuild embeddings for consistent multimodal retrieval")
        logger.info("search engine ready: %d products", len(store))
        return cls(config, store, indexes, encoder)

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.store)

    def threshold(self, mode: str) -> float:
        return self.config.matching.threshold_for(mode)

    def row_of(self, product_id: str) -> int:
        """Catalog row for a product ID.

        Raises:
            KeyError: If the product is not in the catalog.
        """
        try:
            return self._row_by_id[str(product_id)]
        except KeyError:
            raise KeyError(f"unknown product_id: {product_id}") from None

    def product(self, product_id: str) -> dict:
        """Metadata for a product ID."""
        return self.store.metadata.iloc[self.row_of(product_id)].to_dict()

    def _query_vectors(self, image: Image.Image | None, title: str | None,
                       mode: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if mode not in VALID_MODES:
            raise QueryError(f"mode must be one of {VALID_MODES}, got '{mode}'")
        title = (title or "").strip()
        use_image = mode in ("image", "multimodal") and image is not None
        use_text = mode in ("text", "multimodal") and bool(title)
        if mode == "image" and not use_image:
            raise QueryError("image mode requires an image")
        if mode == "text" and not use_text:
            raise QueryError("text mode requires a non-empty title")
        if mode == "multimodal" and not (use_image or use_text):
            raise QueryError("multimodal mode requires an image and/or a title")
        if self.config.data.clean_titles and use_text:
            from src.data.preprocessing import clean_title
            title = clean_title(title) or title.lower()
        batch = self.encoder.encode(images=[image if use_image else None],
                                    texts=[title if use_text else None])
        return (batch.image[0] if use_image else None, batch.text[0] if use_text else None)

    # ------------------------------------------------------------------ search
    def search_vectors(self, q_image: np.ndarray | None, q_text: np.ndarray | None,
                       mode: str, top_k: int | None = None,
                       exclude_rows: Iterable[int] = ()) -> list[SearchResult]:
        """Search with already-projected query vectors.

        Args:
            q_image: Projected query image vector or ``None``.
            q_text: Projected query text vector or ``None``.
            mode: ``image``, ``text`` or ``multimodal``.
            top_k: Number of results (defaults to ``retrieval.top_k``).
            exclude_rows: Catalog rows to drop (e.g. the query product itself).
        """
        top_k = top_k or self.config.retrieval.top_k
        if top_k <= 0:
            raise QueryError("top_k must be positive")
        exclude = set(exclude_rows)
        wi, wt = modality_weights(mode, *self.config.fusion.normalized_weights)
        if mode == "multimodal" and (q_image is None or q_text is None):
            # Only one modality in the query: search that modality's index directly.
            index_name = "image" if q_image is not None else "text"
        else:
            index_name = MODE_TO_INDEX[mode]
        if index_name == "fused":
            f = self.config.fusion
            from src.embeddings.fusion import fuse
            query = fuse(q_image[None], q_text[None], f.method, f.image_weight, f.text_weight)[0]
            n_candidates = top_k * self.config.retrieval.candidate_multiplier
        else:
            query = q_image if index_name == "image" else q_text
            n_candidates = top_k
        _, ids = self.indexes[index_name].search(query, n_candidates + len(exclude))
        rows = np.array([i for i in ids[0] if i >= 0 and i not in exclude], dtype=int)
        if len(rows) == 0:
            return []

        s = self.store
        img_sim, txt_sim, combined, img_ok, txt_ok = score_candidates(
            q_image if mode != "text" else None, q_text if mode != "image" else None,
            s.image[rows], s.text[rows], s.image_mask[rows], s.text_mask[rows], wi, wt)
        # Drop gallery items that lack the modality the query is matched on.
        valid = (img_ok | txt_ok)
        order = [i for i in np.argsort(-combined, kind="stable") if valid[i]][:top_k]
        threshold = self.threshold(mode)
        meta = s.metadata
        results = []
        for rank, i in enumerate(order, start=1):
            r = meta.iloc[rows[i]]
            results.append(SearchResult(
                rank=rank, product_id=str(r["product_id"]), title=str(r["title"]),
                image_path=str(r.get("image_path", "")), image_file=str(r.get("image_file", "")),
                category=str(r.get("category", "")),
                image_similarity=_clean_float(img_sim[i]), text_similarity=_clean_float(txt_sim[i]),
                combined_similarity=round(float(combined[i]), 4),
                is_match=bool(combined[i] >= threshold)))
        return results

    def search(self, image: Image.Image | None = None, title: str | None = None,
               mode: str = "multimodal", top_k: int | None = None) -> list[SearchResult]:
        """Encode a query and return the top-k most similar catalog products."""
        q_image, q_text = self._query_vectors(image, title, mode)
        return self.search_vectors(q_image, q_text, mode, top_k)

    def search_by_image(self, image: Image.Image, top_k: int | None = None) -> list[SearchResult]:
        """Image-only retrieval."""
        return self.search(image=image, mode="image", top_k=top_k)

    def search_by_text(self, title: str, top_k: int | None = None) -> list[SearchResult]:
        """Title-only retrieval."""
        return self.search(title=title, mode="text", top_k=top_k)

    def search_multimodal(self, image: Image.Image | None, title: str | None,
                          top_k: int | None = None) -> list[SearchResult]:
        """Image + title retrieval (falls back to whichever is provided)."""
        return self.search(image=image, title=title, mode="multimodal", top_k=top_k)

    def search_by_product_id(self, product_id: str, mode: str = "multimodal",
                             top_k: int | None = None) -> list[SearchResult]:
        """Product-to-product retrieval using stored embeddings (excludes the product itself)."""
        row = self.row_of(product_id)
        s = self.store
        q_image = s.image[row] if s.image_mask[row] and mode != "text" else None
        q_text = s.text[row] if s.text_mask[row] and mode != "image" else None
        if q_image is None and q_text is None:
            raise QueryError(f"product {product_id} has no usable {mode} embedding")
        return self.search_vectors(q_image, q_text, mode, top_k, exclude_rows=[row])

    # ------------------------------------------------------------------ matching
    def _stored_vectors(self, product_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        row = self.row_of(product_id)
        s = self.store
        return (s.image[row] if s.image_mask[row] else None,
                s.text[row] if s.text_mask[row] else None)

    def match(self, image_a: Image.Image | None = None, title_a: str | None = None,
              image_b: Image.Image | None = None, title_b: str | None = None,
              product_id_a: str | None = None, product_id_b: str | None = None) -> MatchResult:
        """Decide whether two products are the same.

        Each side is given either as a catalog ``product_id`` or as an
        uploaded image and/or title.
        """
        def side(image, title, pid, label):
            if pid:
                return self._stored_vectors(pid)
            if image is None and not (title or "").strip():
                raise QueryError(f"product {label} needs an image, a title or a product_id")
            return self._query_vectors(image, title, "multimodal")

        img_a, txt_a = side(image_a, title_a, product_id_a, "A")
        img_b, txt_b = side(image_b, title_b, product_id_b, "B")
        wi, wt = self.config.fusion.normalized_weights
        return match_pair(img_a, txt_a, img_b, txt_b, wi, wt, self.threshold("multimodal"))
