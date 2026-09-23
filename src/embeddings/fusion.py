"""Embedding normalisation, multimodal fusion and combined similarity scores.

Why L2 normalisation? Once every vector has unit length, the inner product of
two vectors equals their cosine similarity. That makes scores comparable
across products, bounded in [-1, 1] (so a single threshold is meaningful) and
lets FAISS use a fast exact inner-product index for cosine search.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise ``x`` along ``axis``; all-zero rows stay zero (no NaNs)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return np.where(norm > EPS, x / np.maximum(norm, EPS), 0.0).astype(np.float32)


def _mask(mask: np.ndarray | None, n: int) -> np.ndarray:
    return np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def fuse_weighted_sum(image_emb: np.ndarray, text_emb: np.ndarray,
                      image_weight: float, text_weight: float,
                      image_mask: np.ndarray | None = None,
                      text_mask: np.ndarray | None = None) -> np.ndarray:
    """Fuse same-dimensional embeddings as ``normalize(w_i * img + w_t * txt)``.

    If a row lacks one modality (mask ``False``), that modality contributes
    nothing and the result is the normalised remaining modality.

    Raises:
        ValueError: If the embedding shapes differ.
    """
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"fusion needs equal shapes, got {image_emb.shape} vs {text_emb.shape}; "
                         "project both modalities to the same dimension first")
    n = image_emb.shape[0]
    wi = image_weight * _mask(image_mask, n)[:, None]
    wt = text_weight * _mask(text_mask, n)[:, None]
    return l2_normalize(wi * l2_normalize(image_emb) + wt * l2_normalize(text_emb))


def fuse_concat(image_emb: np.ndarray, text_emb: np.ndarray,
                image_mask: np.ndarray | None = None,
                text_mask: np.ndarray | None = None) -> np.ndarray:
    """Fuse by concatenating normalised modality vectors, then re-normalising.

    For two rows that both have both modalities, the inner product of the
    result equals ``(sim_img + sim_txt) / 2`` — an equal-weight score.
    """
    n = image_emb.shape[0]
    img = l2_normalize(image_emb) * _mask(image_mask, n)[:, None]
    txt = l2_normalize(text_emb) * _mask(text_mask, n)[:, None]
    return l2_normalize(np.concatenate([img, txt], axis=1))


def fuse(image_emb: np.ndarray, text_emb: np.ndarray, method: str,
         image_weight: float, text_weight: float,
         image_mask: np.ndarray | None = None,
         text_mask: np.ndarray | None = None) -> np.ndarray:
    """Dispatch to :func:`fuse_weighted_sum` or :func:`fuse_concat`."""
    if method == "weighted_sum":
        return fuse_weighted_sum(image_emb, text_emb, image_weight, text_weight,
                                 image_mask, text_mask)
    if method == "concat":
        return fuse_concat(image_emb, text_emb, image_mask, text_mask)
    raise ValueError(f"unknown fusion method: {method}")


def combined_similarity(image_sim: np.ndarray | float | None,
                        text_sim: np.ndarray | float | None,
                        image_weight: float, text_weight: float,
                        image_available: np.ndarray | bool = True,
                        text_available: np.ndarray | bool = True) -> np.ndarray:
    """Weighted combination of per-modality cosine similarities.

    ``combined = w_i * image_sim + w_t * text_sim``, with weights renormalised
    over the modalities that are available for *both* products of a pair. If a
    pair shares no modality the combined score is 0.

    Args:
        image_sim: Image cosine similarity (scalar or array); ``None`` = unavailable.
        text_sim: Text cosine similarity (scalar or array); ``None`` = unavailable.
        image_weight: Image weight.
        text_weight: Text weight.
        image_available: Whether the image similarity is valid (per element).
        text_available: Whether the text similarity is valid (per element).
    """
    img_ok = np.asarray(image_available, dtype=bool) & (image_sim is not None)
    txt_ok = np.asarray(text_available, dtype=bool) & (text_sim is not None)
    s_i = np.nan_to_num(np.asarray(0.0 if image_sim is None else image_sim, dtype=np.float32))
    s_t = np.nan_to_num(np.asarray(0.0 if text_sim is None else text_sim, dtype=np.float32))
    wi = image_weight * img_ok
    wt = text_weight * txt_ok
    total = wi + wt
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(total > 0, (wi * s_i + wt * s_t) / np.where(total > 0, total, 1.0), 0.0)
    return score.astype(np.float32)
