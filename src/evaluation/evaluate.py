"""End-to-end evaluation and ablation of embedding variants.

Protocol (leakage-safe):

1. Split products into train/test by ``label_group`` (see ``src.data.dataset``).
2. Gallery = the full catalog. Queries = products of one split; each query's
   relevant set is the other listings of its group (always in the same split).
3. For each query, retrieve the top ``candidate_k`` neighbours with FAISS
   (self excluded) -> ranking metrics and candidate pairs.
4. The MATCH threshold is **selected on train queries** and **reported on test
   queries**, so the threshold is never tuned on the data it is scored on.
   Positive pairs that were not retrieved count as false negatives.

Experiments (all computed from the same cached embeddings):

A. image-only             projected CLIP embedding
B. text-only              projected sentence embedding
C. concat                 normalize([img, txt])      (equal weights)
D. weighted fusion        normalize(w_i*img + w_t*txt)
E. weighted score fusion  normalize([sqrt(w_i)*img, sqrt(w_t)*txt]); its inner
                          product equals w_i*sim_img + w_t*sim_txt exactly, i.e.
                          the combined score the matcher and search engine use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import ProductCatalog, split_indices
from src.embeddings.fusion import fuse_concat, fuse_weighted_sum, l2_normalize
from src.embeddings.store import EmbeddingStore
from src.evaluation import metrics as M
from src.evaluation.threshold_analysis import (best_threshold, precision_recall_points,
                                               sweep_thresholds)
from src.retrieval.faiss_index import FaissIndex
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

EXPERIMENTS = {
    "A_image_only": ("Image only", "image"),
    "B_text_only": ("Text only", "text"),
    "C_concat": ("Image + Text concat", "multimodal"),
    "D_weighted_fusion": ("Weighted fusion", "multimodal"),
    "E_weighted_score": ("Weighted score fusion", "multimodal"),
}


def build_experiment_embeddings(store: EmbeddingStore, image_weight: float,
                                text_weight: float) -> dict[str, np.ndarray]:
    """Embedding matrix for each ablation experiment."""
    img, txt, mi, mt = store.image, store.text, store.image_mask, store.text_mask
    wi, wt = image_weight / (image_weight + text_weight), text_weight / (image_weight + text_weight)
    score_level = np.concatenate([np.sqrt(wi) * img * mi[:, None], np.sqrt(wt) * txt * mt[:, None]], 1)
    return {
        "A_image_only": l2_normalize(img),
        "B_text_only": l2_normalize(txt),
        "C_concat": fuse_concat(img, txt, mi, mt),
        "D_weighted_fusion": fuse_weighted_sum(img, txt, wi, wt, mi, mt),
        "E_weighted_score": l2_normalize(score_level),
    }


@dataclass
class QuerySetResult:
    """Retrieval output for one set of queries."""

    query_rows: np.ndarray
    neighbors: np.ndarray          # (n_q, k) gallery rows, self excluded
    scores: np.ndarray             # (n_q, k)
    relevant: list[set[int]]
    pair_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pair_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    total_positives: int = 0


def retrieve(index: FaissIndex, embeddings: np.ndarray, query_rows: np.ndarray,
             labels: np.ndarray, k: int) -> QuerySetResult:
    """Retrieve top-``k`` neighbours (excluding self) and form candidate pairs."""
    k = min(k, index.ntotal - 1)
    raw_scores, raw_ids = index.search(embeddings[query_rows], k + 1)
    neighbors = np.full((len(query_rows), k), -1, dtype=int)
    scores = np.full((len(query_rows), k), -np.inf, dtype=np.float32)
    relevant: list[set[int]] = []
    for qi, row in enumerate(query_rows):
        keep = [(i, s) for i, s in zip(raw_ids[qi], raw_scores[qi]) if i != row and i >= 0][:k]
        neighbors[qi, :len(keep)] = [i for i, _ in keep]
        scores[qi, :len(keep)] = [s for _, s in keep]
        relevant.append(set(np.flatnonzero(labels == labels[row])) - {int(row)})
    valid = neighbors >= 0
    pair_labels = (labels[np.where(valid, neighbors, 0)] == labels[query_rows][:, None]) & valid
    return QuerySetResult(query_rows, neighbors, scores, relevant,
                          pair_scores=scores[valid], pair_labels=pair_labels[valid],
                          total_positives=sum(len(r) for r in relevant))


def ranking_metrics(res: QuerySetResult, k_values: list[int]) -> dict[str, float]:
    """Recall@K, Precision@K, HitRate@K and mAP for a query set."""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = M.recall_at_k(res.neighbors, res.relevant, k)
        out[f"precision@{k}"] = M.precision_at_k(res.neighbors, res.relevant, k)
        out[f"hit_rate@{k}"] = M.hit_rate_at_k(res.neighbors, res.relevant, k)
    out[f"map@{max(k_values)}"] = M.mean_average_precision(res.neighbors, res.relevant, max(k_values))
    return out


def query_f1(res: QuerySetResult, threshold: float) -> float:
    """Mean per-query F1 of the predicted match set at ``threshold``."""
    predicted = [set(n[s >= threshold].tolist()) for n, s in zip(res.neighbors, res.scores)]
    return M.mean_query_f1(predicted, res.relevant)


def run_evaluation(config: Config, store: EmbeddingStore) -> dict:
    """Run all ablation experiments and return a JSON-serialisable report.

    Also writes CSV/Markdown/JSON results to ``paths.results_dir`` and plots
    to ``paths.plots_dir``.
    """
    catalog = ProductCatalog(store.metadata)
    labels = catalog.labels
    train_rows, test_rows = split_indices(catalog, config.split.strategy, config.split.test_size,
                                          config.split.n_splits, config.split.fold, config.seed)
    ev = config.evaluation
    logger.info("evaluating on %d test queries (%d groups); thresholds tuned on %d train queries",
                len(test_rows), len(np.unique(labels[test_rows])), len(train_rows))

    embeddings = build_experiment_embeddings(store, config.fusion.image_weight, config.fusion.text_weight)
    summary_rows, sweeps, pr_curves, details = [], [], {}, {}
    for name, emb in embeddings.items():
        label, mode = EXPERIMENTS[name]
        index = FaissIndex.build(emb, config.retrieval.index_type)
        train = retrieve(index, emb, train_rows, labels, ev.candidate_k)
        test = retrieve(index, emb, test_rows, labels, ev.candidate_k)

        # Threshold selected on train queries only.
        train_sweep = sweep_thresholds(train.pair_scores, train.pair_labels, ev.thresholds,
                                       train.total_positives)
        tuned = best_threshold(train_sweep)["threshold"]
        test_sweep = sweep_thresholds(test.pair_scores, test.pair_labels, ev.thresholds,
                                      test.total_positives)
        at_tuned = test_sweep.loc[np.isclose(test_sweep["threshold"], tuned)].iloc[0]
        config_thr = config.matching.threshold_for(mode)
        at_config = M.classification_metrics(test.pair_labels, test.pair_scores >= config_thr,
                                             test.total_positives)
        rank = ranking_metrics(test, ev.k_values)

        summary_rows.append({
            "experiment": name, "label": label, **rank,
            "tuned_threshold": tuned,
            "precision": at_tuned["precision"], "recall": at_tuned["recall"],
            "f1": at_tuned["f1"], "accuracy": at_tuned["accuracy"],
            "query_f1": query_f1(test, tuned),
            "config_threshold": config_thr, "f1_at_config_threshold": at_config["f1"],
        })
        sweeps.append(test_sweep.assign(experiment=name, label=label))
        p, r, _ = precision_recall_points(test.pair_scores, test.pair_labels, test.total_positives)
        pr_curves[name] = (p, r)
        details[name] = {"train_sweep": train_sweep.to_dict("records")}

    summary = pd.DataFrame(summary_rows)
    sweep_df = pd.concat(sweeps, ignore_index=True)
    report = {
        "n_products": len(store), "n_groups": int(len(np.unique(labels))),
        "n_train_queries": int(len(train_rows)), "n_test_queries": int(len(test_rows)),
        "split_strategy": config.split.strategy, "seed": config.seed,
        "embeddings_manifest": store.manifest,
        "fusion_weights": {"image": config.fusion.image_weight, "text": config.fusion.text_weight},
        "summary": summary.to_dict("records"), "details": details,
    }
    _write_outputs(config, summary, sweep_df, report)

    from src.evaluation import plots
    plots_dir = config.resolve(config.paths.plots_dir)
    plots.plot_threshold_vs_f1(sweep_df, plots_dir / "threshold_vs_f1.png")
    plots.plot_precision_recall(pr_curves, plots_dir / "precision_recall.png")
    plots.plot_recall_at_k(summary, ev.k_values, plots_dir / "recall_at_k.png")
    plots.plot_modality_comparison(summary, max(ev.k_values), plots_dir / "modality_comparison.png")
    logger.info("plots written to %s", plots_dir)
    return report


def _markdown_table(summary: pd.DataFrame, k_values: list[int]) -> str:
    cols = ["label"] + [f"recall@{k}" for k in k_values] + [f"map@{max(k_values)}", "tuned_threshold",
                                                            "precision", "recall", "f1", "query_f1"]
    headers = ["Experiment"] + [f"R@{k}" for k in k_values] + [f"mAP@{max(k_values)}", "Thr (train)",
                                                               "Precision", "Recall", "F1", "Query F1"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in summary.iterrows():
        cells = [r["label"]] + [f"{r[c]:.3f}" if c != "tuned_threshold" else f"{r[c]:.2f}" for c in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_outputs(config: Config, summary: pd.DataFrame, sweep_df: pd.DataFrame, report: dict) -> None:
    out = config.resolve(config.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "ablation.csv", index=False)
    sweep_df.to_csv(out / "threshold_sweep.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=_json_default))
    header = (f"<!-- generated by scripts/evaluate.py; test queries={report['n_test_queries']}, "
              f"split={report['split_strategy']}, seed={report['seed']} -->\n")
    (out / "ablation.md").write_text(header + _markdown_table(summary, config.evaluation.k_values) + "\n")
    logger.info("results written to %s", out)


def _json_default(o: object) -> object:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")
