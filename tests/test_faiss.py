"""FAISS index creation, search, persistence and validation; search engine."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.embeddings.fusion import l2_normalize
from src.retrieval.faiss_index import (FaissIndex, InvalidIndexError, load_index_set,
                                       save_index_set)
from src.retrieval.search import QueryError


@pytest.fixture
def vectors() -> np.ndarray:
    return l2_normalize(np.random.default_rng(1).normal(size=(20, 8)))


def test_build_index(vectors):
    index = FaissIndex.build(vectors)
    assert index.ntotal == 20 and index.dim == 8


def test_search_returns_self_first_with_cosine_scores(vectors):
    index = FaissIndex.build(vectors)
    scores, ids = index.search(vectors[:5], k=3)
    assert ids[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)
    assert np.allclose(scores[0, 1], vectors[0] @ vectors[ids[0, 1]], atol=1e-5)
    assert np.all(np.diff(scores, axis=1) <= 1e-6)  # descending


def test_search_accepts_1d_query_and_clips_k(vectors):
    scores, ids = FaissIndex.build(vectors).search(vectors[0], k=100)
    assert ids.shape == (1, 20)


def test_search_validates_inputs(vectors):
    index = FaissIndex.build(vectors)
    with pytest.raises(ValueError):
        index.search(vectors[:, :4], k=2)
    with pytest.raises(ValueError):
        index.search(vectors, k=0)


def test_build_rejects_bad_embeddings():
    with pytest.raises(ValueError):
        FaissIndex.build(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        FaissIndex.build(np.array([[np.nan, 1.0]]))


def test_save_and_load_roundtrip(tmp_path, vectors):
    FaissIndex.build(vectors).save(tmp_path / "x.faiss")
    loaded = FaissIndex.load(tmp_path / "x.faiss", expected_dim=8, expected_size=20)
    _, ids = loaded.search(vectors[3], 1)
    assert ids[0, 0] == 3


def test_load_invalid_index_raises(tmp_path, vectors):
    with pytest.raises(InvalidIndexError, match="not found"):
        FaissIndex.load(tmp_path / "missing.faiss")
    (tmp_path / "corrupt.faiss").write_bytes(b"not a faiss index")
    with pytest.raises(InvalidIndexError):
        FaissIndex.load(tmp_path / "corrupt.faiss")
    FaissIndex.build(vectors).save(tmp_path / "ok.faiss")
    with pytest.raises(InvalidIndexError, match="dim"):
        FaissIndex.load(tmp_path / "ok.faiss", expected_dim=16)
    with pytest.raises(InvalidIndexError, match="stale"):
        FaissIndex.load(tmp_path / "ok.faiss", expected_size=5)


def test_index_set_detects_catalog_mismatch(tmp_path, vectors):
    ids = [f"P{i}" for i in range(20)]
    save_index_set(tmp_path, {"fused": FaissIndex.build(vectors)}, ids)
    assert load_index_set(tmp_path, ids)["fused"].ntotal == 20
    with pytest.raises(InvalidIndexError, match="different embeddings"):
        load_index_set(tmp_path, ids[::-1])


# ------------------------------------------------------------------ search engine
def test_engine_multimodal_search_finds_same_group(built_engine):
    q = built_engine.product("T001")  # red shirt
    img = Image.open(q["image_file"]).convert("RGB")
    results = built_engine.search(image=img, title=q["title"], mode="multimodal", top_k=3)
    assert results[0].product_id == "T001"
    groups = [built_engine.product(r.product_id)["label_group"] for r in results[:2]]
    assert groups == ["shirt_red", "shirt_red"]
    by_id = {r.product_id: r for r in results}
    if "T900" in by_id:  # listing without an image: image similarity is reported as missing
        assert by_id["T900"].image_similarity is None and by_id["T900"].text_similarity is not None


def test_engine_modes_and_validation(built_engine):
    text_res = built_engine.search_by_text("blue cotton mug brand x", top_k=5)
    assert all(r.image_similarity is None for r in text_res)
    img_res = built_engine.search_by_image(Image.new("RGB", (8, 8), (30, 170, 60)), top_k=3)
    assert all(r.text_similarity is None for r in img_res)
    # the product with an empty title never appears in text results
    assert "T901" not in [r.product_id for r in built_engine.search_by_text("mug", top_k=20)]
    with pytest.raises(QueryError):
        built_engine.search(title="x", mode="image")
    with pytest.raises(QueryError):
        built_engine.search(image=None, title="  ", mode="multimodal")
    with pytest.raises(QueryError):
        built_engine.search(title="x", mode="video")


def test_engine_search_by_product_id_excludes_self(built_engine):
    results = built_engine.search_by_product_id("T003", top_k=5)
    ids = [r.product_id for r in results]
    assert "T003" not in ids
    assert "T004" in ids  # the other blue-shirt listing is retrieved
    assert all(a.combined_similarity >= b.combined_similarity for a, b in zip(results, results[1:]))
