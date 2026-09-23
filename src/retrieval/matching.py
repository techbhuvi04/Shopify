"""Pairwise product scoring and match decisions.

For products A and B::

    image_similarity    = cos(img_A, img_B)
    text_similarity     = cos(txt_A, txt_B)
    combined_similarity = w_i * image_similarity + w_t * text_similarity
    MATCH  if combined_similarity >= threshold  else  NOT_MATCH

Weights and thresholds come from ``configs/config.yaml``; nothing here is
hard-coded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.embeddings.fusion import combined_similarity

MATCH = "MATCH"
NOT_MATCH = "NOT_MATCH"


def apply_threshold(scores: np.ndarray | float, threshold: float) -> np.ndarray:
    """Boolean match decisions: ``scores >= threshold``."""
    return np.asarray(scores) >= threshold


def modality_weights(mode: str, image_weight: float, text_weight: float) -> tuple[float, float]:
    """Effective ``(image_weight, text_weight)`` for a retrieval mode."""
    if mode == "image":
        return 1.0, 0.0
    if mode == "text":
        return 0.0, 1.0
    if mode == "multimodal":
        return image_weight, text_weight
    raise ValueError(f"unknown mode: {mode}")


def score_candidates(q_image: np.ndarray | None, q_text: np.ndarray | None,
                     gallery_image: np.ndarray, gallery_text: np.ndarray,
                     gallery_image_mask: np.ndarray, gallery_text_mask: np.ndarray,
                     image_weight: float, text_weight: float,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score one query against candidate gallery rows.

    Args:
        q_image: Projected query image embedding ``(d,)`` or ``None``.
        q_text: Projected query text embedding ``(d,)`` or ``None``.
        gallery_image: Candidate image embeddings ``(m, d)``.
        gallery_text: Candidate text embeddings ``(m, d)``.
        gallery_image_mask: Which candidates have an image.
        gallery_text_mask: Which candidates have a title.
        image_weight: Image weight for the combined score.
        text_weight: Text weight for the combined score.

    Returns:
        ``(image_sim, text_sim, combined, image_ok, text_ok)``; similarities are
        NaN where the modality is unavailable for the pair.
    """
    m = gallery_image.shape[0]
    image_ok = np.asarray(gallery_image_mask, bool) & (q_image is not None)
    text_ok = np.asarray(gallery_text_mask, bool) & (q_text is not None)
    image_sim = gallery_image @ q_image if q_image is not None else np.zeros(m, np.float32)
    text_sim = gallery_text @ q_text if q_text is not None else np.zeros(m, np.float32)
    combined = combined_similarity(image_sim, text_sim, image_weight, text_weight, image_ok, text_ok)
    return (np.where(image_ok, image_sim, np.nan), np.where(text_ok, text_sim, np.nan),
            combined, image_ok, text_ok)


@dataclass
class MatchResult:
    """Result of comparing two products."""

    image_similarity: float | None
    text_similarity: float | None
    combined_similarity: float
    threshold: float
    is_match: bool

    @property
    def decision(self) -> str:
        return MATCH if self.is_match else NOT_MATCH

    def to_dict(self) -> dict:
        return {**asdict(self), "decision": self.decision}


def match_pair(image_a: np.ndarray | None, text_a: np.ndarray | None,
               image_b: np.ndarray | None, text_b: np.ndarray | None,
               image_weight: float, text_weight: float, threshold: float) -> MatchResult:
    """Compare two products given their projected, normalised embeddings.

    Any embedding may be ``None`` (missing image or empty title); the combined
    score then uses only the modalities both products have.
    """
    image_sim = float(image_a @ image_b) if image_a is not None and image_b is not None else None
    text_sim = float(text_a @ text_b) if text_a is not None and text_b is not None else None
    combined = float(combined_similarity(image_sim, text_sim, image_weight, text_weight,
                                         image_sim is not None, text_sim is not None))
    return MatchResult(image_sim, text_sim, combined, threshold, bool(combined >= threshold))
