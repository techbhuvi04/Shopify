"""FastAPI inference service.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                      service + index status
    POST /search   (multipart)        image?, title?, mode, top_k
    POST /match    (multipart)        product A vs product B
    GET  /products/{product_id}/image catalog image

Models and indexes are loaded once at startup and shared by all requests.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.data.preprocessing import load_image  # noqa: E402
from src.retrieval.search import ProductSearchEngine, QueryError  # noqa: E402
from src.utils.config import VALID_MODES, Config, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TOP_K = 100


# ---------------------------------------------------------------------- schemas
class QueryInfo(BaseModel):
    title: str | None
    mode: str
    has_image: bool
    top_k: int


class ResultItem(BaseModel):
    rank: int
    product_id: str
    title: str
    image_path: str
    category: str
    image_similarity: float | None
    text_similarity: float | None
    combined_similarity: float
    is_match: bool


class SearchResponse(BaseModel):
    query: QueryInfo
    threshold: float
    results: list[ResultItem]


class MatchResponse(BaseModel):
    image_similarity: float | None
    text_similarity: float | None
    combined_similarity: float
    threshold: float
    is_match: bool
    decision: str


class HealthResponse(BaseModel):
    status: str
    products: int
    device: str | None
    detail: str | None = None


# ---------------------------------------------------------------------- app
EngineFactory = Callable[[Config], ProductSearchEngine]


def create_app(config: Config | None = None, engine_factory: EngineFactory | None = None) -> FastAPI:
    """Build the FastAPI app.

    Args:
        config: Configuration (defaults to :func:`load_config`).
        engine_factory: Builds the search engine; injectable for tests.
    """
    config = config or load_config()
    factory = engine_factory or ProductSearchEngine.from_artifacts

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine, app.state.error = None, None
        try:
            app.state.engine = factory(config)
        except Exception as exc:  # keep the service up and report via /health
            logger.exception("failed to load search engine")
            app.state.error = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(title="Multimodal Product Matching API", version="0.1.0", lifespan=lifespan)
    # Torch models are not guaranteed thread-safe; serialise inference.
    lock = threading.Lock()

    def engine(request: Request) -> ProductSearchEngine:
        eng = request.app.state.engine
        if eng is None:
            raise HTTPException(503, f"search engine unavailable: {request.app.state.error}. "
                                     "Build embeddings and the index first.")
        return eng

    def read_upload(upload: UploadFile | None, field: str) -> Image.Image | None:
        if upload is None or not upload.filename:
            return None
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(415, f"{field}: expected an image, got {upload.content_type}")
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{field}: image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        img = load_image(data)
        if img is None:
            raise HTTPException(400, f"{field}: could not decode image")
        return img

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        eng = request.app.state.engine
        if eng is None:
            return HealthResponse(status="degraded", products=0, device=None, detail=request.app.state.error)
        return HealthResponse(status="ok", products=len(eng), device=str(eng.encoder.device))

    @app.post("/search", response_model=SearchResponse)
    def search(request: Request,
               image: UploadFile | None = File(None, description="query product image"),
               title: str | None = Form(None, description="query product title"),
               mode: str = Form("multimodal", description="image | text | multimodal"),
               top_k: int = Form(config.retrieval.top_k, ge=1, le=MAX_TOP_K)) -> SearchResponse:
        eng = engine(request)
        if mode not in VALID_MODES:
            raise HTTPException(422, f"mode must be one of {list(VALID_MODES)}")
        img = read_upload(image, "image")
        try:
            with lock:
                results = eng.search(image=img, title=title, mode=mode, top_k=top_k)
        except QueryError as exc:
            raise HTTPException(422, str(exc)) from exc
        return SearchResponse(
            query=QueryInfo(title=title, mode=mode, has_image=img is not None, top_k=top_k),
            threshold=eng.threshold(mode),
            results=[ResultItem(**r.to_dict()) for r in results])

    @app.post("/match", response_model=MatchResponse)
    def match(request: Request,
              image_a: UploadFile | None = File(None), title_a: str | None = Form(None),
              product_id_a: str | None = Form(None),
              image_b: UploadFile | None = File(None), title_b: str | None = Form(None),
              product_id_b: str | None = Form(None)) -> MatchResponse:
        eng = engine(request)
        img_a, img_b = read_upload(image_a, "image_a"), read_upload(image_b, "image_b")
        try:
            with lock:
                result = eng.match(img_a, title_a, img_b, title_b, product_id_a, product_id_b)
        except QueryError as exc:
            raise HTTPException(422, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0])) from exc
        return MatchResponse(**result.to_dict())

    @app.get("/products/{product_id}/image")
    def product_image(product_id: str, request: Request) -> FileResponse:
        eng = engine(request)
        try:
            path = Path(eng.product(product_id).get("image_file") or "")
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0])) from exc
        if not path.is_file():
            raise HTTPException(404, f"no image for product {product_id}")
        return FileResponse(path)

    return app


app = create_app()
