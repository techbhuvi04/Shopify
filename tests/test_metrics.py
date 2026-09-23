"""Metric calculations, threshold matching and the evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import metrics as M
from src.evaluation.threshold_analysis import (best_threshold, precision_recall_points,
                                               sweep_thresholds)
from src.retrieval.matching import MATCH, NOT_MATCH, apply_threshold, match_pair


def test_classification_metrics_hand_computed():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 1, 0, 1]
    m = M.classification_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_classification_metrics_counts_unretrieved_positives():
    m = M.classification_metrics([1, 0], [1, 0], total_positives=4)
    assert m["recall"] == pytest.approx(0.25) and m["fn"] == 3
    with pytest.raises(ValueError):
        M.classification_metrics([1, 1], [1, 1], total_positives=1)


def test_classification_metrics_zero_division_is_safe():
    m = M.classification_metrics([0, 0], [0, 0])
    assert m["precision"] == m["recall"] == m["f1"] == 0.0 and m["accuracy"] == 1.0


def test_ranking_metrics_hand_computed():
    retrieved = np.array([[5, 1, 2], [7, 8, 9], [3, 4, 6]])
    relevant = [{1, 2}, {9}, set()]  # third query has no relevant item -> skipped
    assert M.recall_at_k(retrieved, relevant, 1) == pytest.approx((0 + 0) / 2)
    assert M.recall_at_k(retrieved, relevant, 2) == pytest.approx((0.5 + 0) / 2)
    assert M.recall_at_k(retrieved, relevant, 3) == pytest.approx((1.0 + 1.0) / 2)
    assert M.precision_at_k(retrieved, relevant, 3) == pytest.approx((2 / 3 + 1 / 3) / 2)
    assert M.hit_rate_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    ap_q1 = (1 / 2 + 2 / 3) / 2
    ap_q2 = (1 / 3) / 1
    assert M.mean_average_precision(retrieved, relevant, 3) == pytest.approx((ap_q1 + ap_q2) / 2)


def test_mean_query_f1():
    assert M.mean_query_f1([{1, 2}, set()], [{1}, set()]) == pytest.approx((2 / 3 + 1.0) / 2)


# ------------------------------------------------------------------ threshold matching
def test_apply_threshold_is_inclusive():
    assert apply_threshold(np.array([0.74, 0.75, 0.9]), 0.75).tolist() == [False, True, True]


def test_match_pair_decision_and_missing_modality():
    a_img, a_txt = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    b_img, b_txt = np.array([1.0, 0.0]), np.array([0.6, 0.8])
    r = match_pair(a_img, a_txt, b_img, b_txt, 0.6, 0.4, threshold=0.9)
    assert r.combined_similarity == pytest.approx(0.6 * 1.0 + 0.4 * 0.8)
    assert r.is_match and r.decision == MATCH
    r2 = match_pair(a_img, a_txt, b_img, b_txt, 0.6, 0.4, threshold=0.95)
    assert r2.decision == NOT_MATCH
    r3 = match_pair(None, a_txt, b_img, b_txt, 0.6, 0.4, threshold=0.5)
    assert r3.image_similarity is None and r3.combined_similarity == pytest.approx(0.8)


def test_threshold_sweep_and_best_threshold():
    scores = np.array([0.95, 0.9, 0.8, 0.7, 0.6])
    labels = np.array([True, True, False, True, False])
    sweep = sweep_thresholds(scores, labels, [0.5, 0.75, 0.85, 0.99])
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep.loc[sweep.threshold == 0.99, "n_predicted"].item() == 0
    best = best_threshold(sweep)
    assert best["threshold"] == 0.85 and best["f1"] == pytest.approx(0.8)


def test_precision_recall_points():
    p, r, t = precision_recall_points(np.array([0.9, 0.8, 0.8, 0.1]),
                                      np.array([True, False, True, False]), total_positives=4)
    assert t.tolist() == pytest.approx([0.9, 0.8, 0.1])
    assert p.tolist() == pytest.approx([1.0, 2 / 3, 0.5])
    assert r.tolist() == pytest.approx([0.25, 0.5, 0.5])


# ------------------------------------------------------------------ evaluation pipeline
def test_run_evaluation_writes_results_and_plots(built_engine, test_config):
    from src.evaluation.evaluate import EXPERIMENTS, run_evaluation
    report = run_evaluation(test_config, built_engine.store)
    assert [row["experiment"] for row in report["summary"]] == list(EXPERIMENTS)
    for row in report["summary"]:
        assert 0.0 <= row["recall@5"] <= 1.0 and 0.0 <= row["f1"] <= 1.0
    results = test_config.resolve(test_config.paths.results_dir)
    plots = test_config.resolve(test_config.paths.plots_dir)
    assert (results / "ablation.md").is_file() and (results / "metrics.json").is_file()
    assert len(list(plots.glob("*.png"))) == 4
