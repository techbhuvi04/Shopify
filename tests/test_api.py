"""FastAPI endpoints with an offline search engine."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app


def png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(built_engine, test_config):
    app = create_app(test_config, engine_factory=lambda _cfg: built_engine)
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["products"] == 18


def test_search_multimodal(client):
    r = client.post("/search", data={"title": "red cotton shirt brand x", "mode": "multimodal", "top_k": "3"},
                    files={"image": ("q.png", png_bytes((210, 30, 30)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query"] == {"title": "red cotton shirt brand x", "mode": "multimodal",
                             "has_image": True, "top_k": 3}
    assert len(body["results"]) == 3
    top = body["results"][0]
    assert {"product_id", "title", "image_similarity", "text_similarity",
            "combined_similarity", "is_match"} <= set(top)
    assert top["product_id"] in {"T001", "T002", "T900"}


def test_search_text_only(client):
    r = client.post("/search", data={"title": "green mug", "mode": "text", "top_k": "2"})
    assert r.status_code == 200 and all(x["image_similarity"] is None for x in r.json()["results"])


@pytest.mark.parametrize("data,files,status", [
    ({"mode": "image"}, None, 422),                                   # image mode without image
    ({"mode": "video", "title": "x"}, None, 422),                      # invalid mode
    ({"mode": "text", "title": "x", "top_k": "0"}, None, 422),         # invalid top_k
    ({"mode": "image"}, {"image": ("x.png", b"garbage", "image/png")}, 400),
    ({"mode": "image"}, {"image": ("x.txt", b"hello", "text/plain")}, 415),
])
def test_search_validation(client, data, files, status):
    assert client.post("/search", data=data, files=files).status_code == status


def test_match_by_ids_and_upload(client):
    r = client.post("/match", data={"product_id_a": "T001", "product_id_b": "T002"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"MATCH", "NOT_MATCH"}
    assert body["is_match"] == (body["combined_similarity"] >= body["threshold"])
    r = client.post("/match", data={"product_id_a": "T001", "title_b": "blue mug"},
                    files={"image_b": ("b.png", png_bytes((30, 60, 210)), "image/png")})
    assert r.status_code == 200
    assert client.post("/match", data={"product_id_a": "NOPE", "product_id_b": "T001"}).status_code == 404
    assert client.post("/match", data={"product_id_a": "T001"}).status_code == 422


def test_product_image(client):
    assert client.get("/products/T001/image").status_code == 200
    assert client.get("/products/T900/image").status_code == 404  # missing image file


def test_degraded_when_artifacts_missing(test_config):
    def failing(_cfg):
        raise FileNotFoundError("no index")
    with TestClient(create_app(test_config, engine_factory=failing)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/search", data={"title": "x", "mode": "text"}).status_code == 503
