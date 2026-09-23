"""Encode the product catalog into image, text and fused embeddings.

Usage:
    python scripts/build_embeddings.py [--config configs/config.yaml]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import time  # noqa: E402

from src.data.dataset import load_catalog_from_config  # noqa: E402
from src.embeddings.generate_image_embeddings import generate_image_embeddings  # noqa: E402
from src.embeddings.generate_text_embeddings import generate_text_embeddings  # noqa: E402
from src.embeddings.store import EmbeddingStore, embeddings_dir  # noqa: E402
from src.models.multimodal_encoder import MultimodalEncoder, save_projection_heads  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("build_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build catalog embeddings.")
    parser.add_argument("--config", type=Path, default=None, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    catalog = load_catalog_from_config(config)
    encoder = MultimodalEncoder(config)

    t0 = time.perf_counter()
    image_raw, image_mask = generate_image_embeddings(catalog, encoder)
    text_raw, text_mask = generate_text_embeddings(catalog, encoder)
    image_head, text_head = encoder.init_heads(image_raw.shape[1], text_raw.shape[1])
    batch = encoder.project(image_raw, text_raw, image_mask, text_mask)
    elapsed = time.perf_counter() - t0

    manifest = {
        "image_model": config.model.image_model,
        "text_model": config.model.text_model,
        "image_raw_dim": int(image_raw.shape[1]),
        "text_raw_dim": int(text_raw.shape[1]),
        "projection_dim": int(batch.image.shape[1]),
        "fused_dim": int(batch.fused.shape[1]),
        "projection_init": config.model.projection_init,
        "projection_weights": config.model.projection_weights,
        "fusion": {"method": config.fusion.method, "image_weight": config.fusion.image_weight,
                   "text_weight": config.fusion.text_weight},
        "n_products": len(catalog),
        "n_missing_images": int((~image_mask).sum()),
        "n_empty_titles": int((~text_mask).sum()),
        "products_csv": config.data.products_csv,
        "device": str(encoder.device),
        "encode_seconds": round(elapsed, 2),
    }
    store = EmbeddingStore(metadata=catalog.df, image_raw=image_raw, text_raw=text_raw,
                           image=batch.image, text=batch.text, fused=batch.fused,
                           image_mask=image_mask, text_mask=text_mask, manifest=manifest)
    out_dir = embeddings_dir(config)
    store.save(out_dir)
    save_projection_heads(out_dir / "projection_heads.pt", image_head, text_head,
                          meta={k: manifest[k] for k in ("image_model", "text_model")})
    logger.info("saved %d embeddings (fused dim %d) to %s in %.1fs",
                len(store), manifest["fused_dim"], out_dir, elapsed)


if __name__ == "__main__":
    main()
