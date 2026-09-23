"""Normalisation, fusion, projection heads and the multimodal encoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.fusion import (combined_similarity, fuse, fuse_concat, fuse_weighted_sum,
                                   l2_normalize)
from src.models.multimodal_encoder import (ProjectionHead, load_projection_heads,
                                           save_projection_heads)

rng = np.random.default_rng(0)


def unit_rows(n: int, d: int) -> np.ndarray:
    return l2_normalize(rng.normal(size=(n, d)))


# ------------------------------------------------------------------ normalisation
def test_l2_normalize_gives_unit_norm():
    x = rng.normal(size=(5, 8)) * 10
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0, atol=1e-6)


def test_l2_normalize_keeps_zero_rows_zero_without_nan():
    x = np.zeros((2, 4))
    x[0] = [3, 4, 0, 0]
    out = l2_normalize(x)
    assert np.allclose(out[0], [0.6, 0.8, 0, 0])
    assert np.all(out[1] == 0) and np.isfinite(out).all()


def test_l2_normalize_returns_float32():
    assert l2_normalize(np.ones((2, 3), dtype=np.float64)).dtype == np.float32


# ------------------------------------------------------------------ fusion
def test_weighted_sum_fusion_is_normalized_and_weighted():
    img, txt = unit_rows(4, 16), unit_rows(4, 16)
    fused = fuse_weighted_sum(img, txt, 0.6, 0.4)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0, atol=1e-6)
    expected = l2_normalize(0.6 * img + 0.4 * txt)
    assert np.allclose(fused, expected, atol=1e-6)


def test_weighted_sum_fusion_rejects_mismatched_dims():
    with pytest.raises(ValueError, match="same dimension"):
        fuse_weighted_sum(unit_rows(2, 8), unit_rows(2, 16), 0.5, 0.5)


def test_fusion_falls_back_to_available_modality():
    img, txt = unit_rows(3, 8), unit_rows(3, 8)
    image_mask = np.array([True, False, True])
    text_mask = np.array([True, True, False])
    fused = fuse_weighted_sum(img, txt, 0.6, 0.4, image_mask, text_mask)
    assert np.allclose(fused[1], txt[1], atol=1e-6)   # no image -> pure text
    assert np.allclose(fused[2], img[2], atol=1e-6)   # no text  -> pure image


def test_concat_inner_product_is_mean_of_modal_similarities():
    img, txt = unit_rows(2, 8), unit_rows(2, 8)
    fused = fuse_concat(img, txt)
    assert fused.shape == (2, 16)
    expected = (img[0] @ img[1] + txt[0] @ txt[1]) / 2
    assert np.isclose(fused[0] @ fused[1], expected, atol=1e-6)


def test_fuse_dispatch_rejects_unknown_method():
    with pytest.raises(ValueError):
        fuse(unit_rows(1, 4), unit_rows(1, 4), "average", 0.5, 0.5)


def test_combined_similarity_weights_and_missing_modalities():
    s = combined_similarity(np.array([0.9, 0.9, 0.9]), np.array([0.5, 0.5, 0.5]), 0.6, 0.4,
                            image_available=np.array([True, False, True]),
                            text_available=np.array([True, True, False]))
    assert np.allclose(s, [0.6 * 0.9 + 0.4 * 0.5, 0.5, 0.9], atol=1e-6)
    assert combined_similarity(None, None, 0.6, 0.4) == 0.0


# ------------------------------------------------------------------ projection heads
def test_projection_head_output_is_normalized_and_deterministic():
    x = rng.normal(size=(6, 32)).astype(np.float32)
    a = ProjectionHead(32, 16, seed=7).project(x)
    b = ProjectionHead(32, 16, seed=7).project(x)
    assert a.shape == (6, 16)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(a, b)


def test_orthogonal_projection_preserves_cosine_when_projecting_up():
    x = unit_rows(5, 8)
    y = ProjectionHead(8, 16, init="orthogonal", seed=1).project(x)
    assert np.allclose(x @ x.T, y @ y.T, atol=1e-5)


def test_projection_head_masks_missing_rows():
    out = ProjectionHead(8, 4).project(unit_rows(3, 8), mask=np.array([True, False, True]))
    assert np.all(out[1] == 0)


def test_projection_heads_roundtrip(tmp_path):
    img, txt = ProjectionHead(12, 8, seed=1), ProjectionHead(20, 8, seed=2)
    save_projection_heads(tmp_path / "heads.pt", img, txt, meta={"k": "v"})
    img2, txt2, meta = load_projection_heads(tmp_path / "heads.pt")
    x = unit_rows(3, 12)
    assert np.allclose(img.project(x), img2.project(x))
    assert meta == {"k": "v"} and txt2.in_dim == 20


# ------------------------------------------------------------------ encoder
def test_multimodal_encoder_shapes_and_masks(fake_encoder):
    from PIL import Image
    images = [Image.new("RGB", (8, 8), (255, 0, 0)), None, Image.new("RGB", (8, 8), (0, 0, 255))]
    batch = fake_encoder.encode(images, ["red shirt", "blue mug", ""])
    assert batch.image.shape == batch.text.shape == batch.fused.shape == (3, 16)
    assert batch.image_mask.tolist() == [True, False, True]
    assert batch.text_mask.tolist() == [True, True, False]
    assert np.all(batch.image[1] == 0) and np.all(batch.text[2] == 0)
    assert np.allclose(np.linalg.norm(batch.fused, axis=1), 1.0, atol=1e-5)
