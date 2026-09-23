"""Shared fixtures: offline fake backbones and a tiny end-to-end pipeline.

Tests never download models. The fake backbones are deterministic functions
of the input (image colour statistics / hashed character trigrams), so similar
inputs get similar vectors and retrieval behaves sensibly.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import load_catalog
from src.embeddings.fusion import l2_normalize
from src.embeddings.generate_image_embeddings import generate_image_embeddings
from src.embeddings.generate_text_embeddings import generate_text_embeddings
from src.embeddings.store import EmbeddingStore
from src.models.multimodal_encoder import MultimodalEncoder, save_projection_heads
from src.retrieval.faiss_index import FaissIndex, save_index_set
from src.retrieval.search import ProductSearchEngine
from src.utils.config import Config, DataConfig, PathsConfig, load_config


class FakeImageBackbone:
    """12-d embedding from a 2x2 downsample of the image."""

    dim = 12

    def encode(self, images: Sequence[Image.Image | None]) -> tuple[np.ndarray, np.ndarray]:
        emb = np.zeros((len(images), self.dim), np.float32)
        mask = np.array([i is not None for i in images])
        for k, img in enumerate(images):
            if img is not None:
                emb[k] = np.asarray(img.resize((2, 2)), np.float32).ravel() / 255.0 - 0.5
        return l2_normalize(emb), mask


class FakeTextBackbone:
    """32-d hashed character-trigram embedding."""

    dim = 32

    def encode(self, texts: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
        emb = np.zeros((len(texts), self.dim), np.float32)
        mask = np.array([bool(t and str(t).strip()) for t in texts])
        for k, t in enumerate(texts):
            s = f"  {str(t or '').lower()} "
            for i in range(len(s) - 2):
                emb[k, zlib.crc32(s[i:i + 3].encode()) % self.dim] += 1.0
        return l2_normalize(emb), mask


COLORS = {"red": (210, 30, 30), "blue": (30, 60, 210), "green": (30, 170, 60), "yellow": (230, 200, 30)}


@pytest.fixture
def tiny_catalog_csv(tmp_path: Path) -> Path:
    """8 groups x 2 listings, plus one row with a missing image and one with an empty title."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    pid = 0
    for noun in ("shirt", "mug"):
        for color, rgb in COLORS.items():
            for variant in range(2):
                pid += 1
                shade = tuple(int(c * (0.95 + 0.05 * variant)) for c in rgb)
                Image.new("RGB", (32, 32), shade).save(img_dir / f"{pid}.png")
                title = f"{color} cotton {noun} brand x" if variant == 0 else f"{noun} {color} brand x original"
                rows.append({"product_id": f"T{pid:03d}", "title": title, "image_path": f"images/{pid}.png",
                             "category": noun, "label_group": f"{noun}_{color}"})
    rows.append({"product_id": "T900", "title": "red cotton shirt brand x", "image_path": "images/missing.png",
                 "category": "shirt", "label_group": "shirt_red"})
    Image.new("RGB", (32, 32), COLORS["blue"]).save(img_dir / "notitle.png")
    rows.append({"product_id": "T901", "title": "", "image_path": "images/notitle.png",
                 "category": "mug", "label_group": "mug_blue"})
    path = tmp_path / "products.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def test_config(tmp_path: Path, tiny_catalog_csv: Path) -> Config:
    """Project config pointed at temporary data and artifact directories."""
    config = load_config()
    config.device = "cpu"
    config.model.projection_dim = 16
    config.model.projection_weights = None
    config.data = DataConfig(products_csv=str(tiny_catalog_csv), image_root=str(tmp_path), clean_titles=True)
    config.paths = PathsConfig(**{k: str(tmp_path / "artifacts" / k) for k in PathsConfig().__dict__})
    config.evaluation.candidate_k = 10
    config.split.test_size = 0.5
    return config


@pytest.fixture
def fake_encoder(test_config: Config) -> MultimodalEncoder:
    return MultimodalEncoder(test_config, image_backbone=FakeImageBackbone(),
                             text_backbone=FakeTextBackbone())


@pytest.fixture
def built_engine(test_config: Config, fake_encoder: MultimodalEncoder) -> ProductSearchEngine:
    """Run the full offline pipeline: catalog -> embeddings -> FAISS -> engine."""
    catalog = load_catalog(test_config.data.products_csv, test_config.data.image_root)
    img_raw, img_mask = generate_image_embeddings(catalog, fake_encoder)
    txt_raw, txt_mask = generate_text_embeddings(catalog, fake_encoder)
    image_head, text_head = fake_encoder.init_heads(img_raw.shape[1], txt_raw.shape[1])
    batch = fake_encoder.project(img_raw, txt_raw, img_mask, txt_mask)
    store = EmbeddingStore(catalog.df, img_raw, txt_raw, batch.image, batch.text, batch.fused,
                           img_mask, txt_mask, manifest={"fusion": {"method": "weighted_sum"}})
    emb_dir = test_config.resolve(test_config.paths.embeddings_dir)
    store.save(emb_dir)
    save_projection_heads(emb_dir / "projection_heads.pt", image_head, text_head)
    indexes = {n: FaissIndex.build(getattr(store, n)) for n in ("image", "text", "fused")}
    save_index_set(test_config.resolve(test_config.paths.indexes_dir), indexes, catalog.product_ids)
    return ProductSearchEngine.from_artifacts(test_config, encoder=fake_encoder)
