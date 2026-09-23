"""Batch generation of raw title embeddings for a catalog."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.data.dataset import ProductCatalog
from src.models.multimodal_encoder import MultimodalEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_text_embeddings(catalog: ProductCatalog, encoder: MultimodalEncoder,
                             chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Encode every (cleaned) catalog title with the text backbone.

    Empty titles yield zero rows and ``mask=False``.

    Returns:
        ``(raw_embeddings, mask)`` aligned with catalog rows.
    """
    titles = catalog.df["clean_title"].tolist()
    parts, masks = [], []
    for start in tqdm(range(0, len(titles), chunk_size), desc="text embeddings", unit="chunk"):
        emb, mask = encoder.encode_texts_raw(titles[start:start + chunk_size])
        parts.append(emb)
        masks.append(mask)
    emb, mask = np.concatenate(parts), np.concatenate(masks)
    if (~mask).any():
        logger.warning("%d titles were empty", int((~mask).sum()))
    return emb, mask
