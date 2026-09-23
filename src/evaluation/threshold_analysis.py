"""Threshold sweeps and precision-recall curves for match decisions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.retrieval.matching import apply_threshold


def sweep_thresholds(scores: np.ndarray, labels: np.ndarray, thresholds: Sequence[float],
                     total_positives: int | None = None) -> pd.DataFrame:
    """Evaluate match decisions at each threshold.

    Args:
        scores: Similarity score per candidate pair.
        labels: ``True`` where the pair is a real match.
        thresholds: Thresholds to evaluate.
        total_positives: See :func:`classification_metrics`.

    Returns:
        DataFrame with one row per threshold and columns ``threshold,
        precision, recall, f1, accuracy, tp, fp, fn, n_predicted``.
    """
    rows = []
    for t in thresholds:
        pred = apply_threshold(scores, t)
        m = classification_metrics(labels, pred, total_positives)
        rows.append({"threshold": float(t), **m, "n_predicted": int(pred.sum())})
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame, metric: str = "f1") -> dict:
    """Row of ``sweep`` maximising ``metric`` (ties -> higher threshold)."""
    if sweep.empty:
        raise ValueError("empty threshold sweep")
    best = sweep.sort_values([metric, "threshold"], ascending=[False, False]).iloc[0]
    return best.to_dict()


def precision_recall_points(scores: np.ndarray, labels: np.ndarray,
                            total_positives: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall at every distinct score cut-off (descending).

    Recall is divided by ``total_positives`` so positives outside the
    candidate set cap the achievable recall, as they would in production.

    Returns:
        ``(precision, recall, thresholds)`` arrays.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    # keep the last index of each run of equal scores
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1] if len(s) else np.array([], int)
    positives = int(labels.sum()) if total_positives is None else int(total_positives)
    precision = tp[last] / np.maximum(tp[last] + fp[last], 1)
    recall = tp[last] / max(positives, 1)
    return precision, recall, s[last]
