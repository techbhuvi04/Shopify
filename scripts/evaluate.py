"""Run retrieval + matching evaluation and the ablation study.

Usage:
    python scripts/evaluate.py [--config configs/config.yaml]

Outputs:
    artifacts/results/ablation.{csv,md}, threshold_sweep.csv, metrics.json
    artifacts/plots/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from src.embeddings.store import EmbeddingStore, embeddings_dir  # noqa: E402
from src.evaluation.evaluate import run_evaluation  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and matching.")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    store = EmbeddingStore.load(embeddings_dir(config))
    report = run_evaluation(config, store)
    md = (config.resolve(config.paths.results_dir) / "ablation.md").read_text()
    print(f"\nTest queries: {report['n_test_queries']}  |  products: {report['n_products']}  "
          f"|  groups: {report['n_groups']}\n")
    print(md)


if __name__ == "__main__":
    main()
