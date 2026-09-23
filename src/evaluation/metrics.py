"""Classification and retrieval metrics for product matching.

Definitions (per query ``q`` with relevant set ``R_q`` = other listings of the
same ``label_group``, and ranked retrieval list ``L_q`` excluding ``q`` itself):

* ``Recall@K``    = mean_q |R_q ∩ L_q[:K]| / |R_q|
* ``Precision@K`` = mean_q |R_q ∩ L_q[:K]| / K
  (upper-bounded by |R_q| / K, so it is low by construction when groups are small)
* ``HitRate@K``   = fraction of queries with at least one relevant item in the top K
* ``mAP@K``       = mean average precision over the top K
* pairwise P/R/F1 treat every (query, candidate) pair as a binary MATCH decision.

Queries without any relevant item in the gallery are skipped for ranking metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def classification_metrics(y_true: Sequence[bool] | np.ndarray, y_pred: Sequence[bool] | np.ndarray,
                           total_positives: int | None = None) -> dict[str, float]:
    """Precision, recall, F1 and accuracy for binary match decisions.

    Args:
        y_true: Ground-truth labels (``True`` = same product).
        y_pred: Predicted labels.
        total_positives: Total number of true matching pairs in the full
            problem. Use this when ``y_true`` covers only a candidate subset
            (e.g. top-K neighbours) so positives that were never retrieved still
            count as false negatives. Defaults to ``sum(y_true)``.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``accuracy``, ``tp``, ``fp``, ``fn``.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    positives = int(t.sum()) if total_positives is None else int(total_positives)
    if positives < int(t.sum()):
        raise ValueError("total_positives cannot be smaller than the positives in y_true")
    fn = positives - tp
    tn = int((~t & ~p).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, positives)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn}


def _hits(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> list[tuple[int, int]]:
    """(hits in top-k, |relevant|) for queries with at least one relevant item."""
    out = []
    for row, rel in zip(retrieved, relevant):
        if rel:
            out.append((sum(1 for r in row[:k] if r in rel), len(rel)))
    return out


def recall_at_k(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """Mean fraction of each query's relevant items found in its top ``k``.

    Args:
        retrieved: ``(n_queries, >=k)`` ranked gallery row ids (self excluded).
        relevant: Per query, the set of relevant gallery row ids.
        k: Cut-off.
    """
    h = _hits(retrieved, relevant, k)
    return float(np.mean([hits / n_rel for hits, n_rel in h])) if h else 0.0


def precision_at_k(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """Mean fraction of the top ``k`` results that are relevant."""
    h = _hits(retrieved, relevant, k)
    return float(np.mean([hits / k for hits, _ in h])) if h else 0.0


def hit_rate_at_k(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """Fraction of queries with at least one relevant item in the top ``k``."""
    h = _hits(retrieved, relevant, k)
    return float(np.mean([hits > 0 for hits, _ in h])) if h else 0.0


def mean_average_precision(retrieved: np.ndarray, relevant: Sequence[set[int]], k: int) -> float:
    """mAP@k, normalised by ``min(|relevant|, k)``."""
    aps = []
    for row, rel in zip(retrieved, relevant):
        if not rel:
            continue
        hits, precisions = 0, []
        for rank, r in enumerate(row[:k], start=1):
            if r in rel:
                hits += 1
                precisions.append(hits / rank)
        aps.append(sum(precisions) / min(len(rel), k))
    return float(np.mean(aps)) if aps else 0.0


def mean_query_f1(predicted: Sequence[set[int]], relevant: Sequence[set[int]]) -> float:
    """Mean per-query F1 between predicted and true match sets.

    This is the row-wise F1 used by marketplace matching benchmarks. A query
    with no true matches and no predictions scores 1.0.
    """
    scores = []
    for pred, rel in zip(predicted, relevant):
        if not pred and not rel:
            scores.append(1.0)
            continue
        tp = len(pred & rel)
        scores.append(_safe_div(2 * tp, len(pred) + len(rel)))
    return float(np.mean(scores)) if scores else 0.0
