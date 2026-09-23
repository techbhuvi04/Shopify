"""Batch generation of raw image embeddings for a catalog."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.data.dataset import ProductCatalog
from src.data.preprocessing import load_image
from src.models.multimodal_encoder import MultimodalEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_image_embeddings(catalog: ProductCatalog, encoder: MultimodalEncoder,
                              chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Encode every catalog image with the image backbone.

    Images are read from disk in chunks so memory stays bounded for large
    catalogs. Missing or unreadable images yield zero rows and ``mask=False``.

    Returns:
        ``(raw_embeddings, mask)`` aligned with catalog rows.
    """
    files = catalog.df["image_file"].tolist()
    parts, masks = [], []
    for start in tqdm(range(0, len(files), chunk_size), desc="image embeddings", unit="chunk"):
        images = [load_image(f) if f else None for f in files[start:start + chunk_size]]
        emb, mask = encoder.encode_images_raw(images)
        parts.append(emb)
        masks.append(mask)
    emb, mask = np.concatenate(parts), np.concatenate(masks)
    if (~mask).any():
        logger.warning("%d images could not be encoded (missing/corrupt)", int((~mask).sum()))
    return emb, mask
