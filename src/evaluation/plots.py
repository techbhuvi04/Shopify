"""Evaluation plots (static PNGs for the README and reports).

Colour follows the experiment, never its rank: each experiment always gets the
same categorical slot. Every series also has its own marker so identity is
never carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Fixed categorical order (validated palette, slots 1-5).
SERIES = {
    "A_image_only": ("Image only", "#2a78d6", "o"),
    "B_text_only": ("Text only", "#eb6834", "s"),
    "C_concat": ("Concat", "#1baf7a", "^"),
    "D_weighted_fusion": ("Weighted fusion", "#eda100", "D"),
    "E_weighted_score": ("Weighted score fusion", "#e87ba4", "v"),
}


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _figure(width: float = 7.5, height: float = 4.5) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _legend(ax: plt.Axes, **kw) -> None:
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY, **kw)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_vs_f1(sweep: pd.DataFrame, path: Path) -> None:
    """F1 on test queries as a function of the match threshold, per experiment."""
    fig, ax = _figure()
    for name, (label, color, marker) in SERIES.items():
        d = sweep[sweep["experiment"] == name]
        if d.empty:
            continue
        ax.plot(d["threshold"], d["f1"], color=color, marker=marker, markersize=6,
                linewidth=2, label=label)
    _style(ax, "Match F1 vs. similarity threshold (test queries)", "Threshold", "F1")
    ax.set_ylim(0, 1)
    _legend(ax, loc="best")
    _save(fig, path)


def plot_precision_recall(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    """Precision-recall curves over candidate pairs (test queries)."""
    fig, ax = _figure(6.5, 5)
    for name, (label, color, marker) in SERIES.items():
        if name not in curves:
            continue
        p, r = curves[name]
        ax.plot(r, p, color=color, linewidth=2, label=label, marker=marker, markersize=6,
                markevery=max(len(r) // 8, 1))
    _style(ax, "Precision vs. recall (test queries)", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _legend(ax, loc="upper right")
    _save(fig, path)


def _grouped_bars(ax: plt.Axes, groups: list[str], series: list[tuple[str, str, list[float]]]) -> None:
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, color, values) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, values, width=width * 0.9, color=color, label=label,
               edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, groups)


def plot_recall_at_k(summary: pd.DataFrame, k_values: list[int], path: Path) -> None:
    """Grouped bars of Recall@K for every experiment."""
    fig, ax = _figure()
    series = []
    for name, (label, color, _) in SERIES.items():
        row = summary[summary["experiment"] == name]
        if not row.empty:
            series.append((label, color, [float(row[f"recall@{k}"].iloc[0]) for k in k_values]))
    _grouped_bars(ax, [f"Recall@{k}" for k in k_values], series)
    _style(ax, "Retrieval recall by experiment (test queries)", "", "Recall")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    _legend(ax, loc="upper left", ncol=3)
    _save(fig, path)


def plot_modality_comparison(summary: pd.DataFrame, max_k: int, path: Path) -> None:
    """Image vs. text vs. multimodal on the headline metrics."""
    fig, ax = _figure()
    metrics = [("recall@1", "Recall@1"), (f"recall@{min(5, max_k)}", f"Recall@{min(5, max_k)}"),
               (f"map@{max_k}", f"mAP@{max_k}"), ("f1", "Match F1"), ("query_f1", "Query F1")]
    metrics = [(c, l) for c, l in metrics if c in summary.columns]
    series = []
    for name in ("A_image_only", "B_text_only", "E_weighted_score"):
        label, color, _ = SERIES[name]
        row = summary[summary["experiment"] == name]
        if not row.empty:
            label = "Multimodal (weighted score)" if name == "E_weighted_score" else label
            series.append((label, color, [float(row[c].iloc[0]) for c, _ in metrics]))
    _grouped_bars(ax, [l for _, l in metrics], series)
    _style(ax, "Image vs. text vs. multimodal (test queries)", "", "Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    _legend(ax, loc="upper right", ncol=3)
    _save(fig, path)
