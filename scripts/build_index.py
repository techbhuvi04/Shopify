"""Build FAISS indexes (image, text, fused) from saved embeddings.

Usage:
    python scripts/build_index.py [--config configs/config.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from src.embeddings.store import EmbeddingStore, embeddings_dir  # noqa: E402
from src.retrieval.faiss_index import FaissIndex, save_index_set  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indexes.")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    store = EmbeddingStore.load(embeddings_dir(config))
    indexes = {name: FaissIndex.build(getattr(store, name), config.retrieval.index_type)
               for name in ("image", "text", "fused")}
    out_dir = config.resolve(config.paths.indexes_dir)
    save_index_set(out_dir, indexes, store.metadata["product_id"].tolist(),
                   extra={"embeddings_manifest": store.manifest})
    for name, idx in indexes.items():
        logger.info("%-6s index: %d vectors, dim %d", name, idx.ntotal, idx.dim)
    logger.info("saved indexes to %s", out_dir)


if __name__ == "__main__":
    main()
