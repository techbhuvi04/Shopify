"""Projection heads and the end-to-end multimodal encoder.

Pipeline per modality::

    backbone embedding (CLIP 512-d / MiniLM 384-d)
        -> linear projection (projection_dim, default 256)
        -> L2 normalisation

and then weighted fusion of the two projected vectors.

The projection heads are needed because the two backbones produce vectors of
different sizes, which cannot be summed. Without training, the heads are
initialised as a *seeded semi-orthogonal* matrix: an orthogonal map roughly
preserves inner products (exactly, when projecting up), so the zero-shot
baseline keeps the backbones' similarity structure. The optional
metric-learning module fine-tunes exactly these heads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.embeddings.fusion import fuse
from src.utils.config import Config
from src.utils.logging import get_logger
from src.utils.seed import resolve_device

logger = get_logger(__name__)


class Backbone(Protocol):
    """Interface shared by the image and text backbones (and test doubles)."""

    dim: int

    def encode(self, items: Sequence) -> tuple[np.ndarray, np.ndarray]: ...


class ProjectionHead(nn.Module):
    """Bias-free linear projection followed by L2 normalisation.

    Args:
        in_dim: Backbone embedding size.
        out_dim: Target embedding size.
        init: ``orthogonal`` (similarity-preserving) or ``random`` (Kaiming).
        seed: Seed for a dedicated RNG so initialisation is reproducible and
            independent of global RNG state.
    """

    def __init__(self, in_dim: int, out_dim: int, init: str = "orthogonal", seed: int = 0) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            if init == "orthogonal":
                nn.init.orthogonal_(self.linear.weight, generator=gen)
            elif init == "random":
                nn.init.kaiming_uniform_(self.linear.weight, generator=gen)
            else:
                raise ValueError(f"unknown projection init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.linear(x), dim=-1)

    @torch.no_grad()
    def project(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Project a NumPy batch; rows with ``mask == False`` stay zero."""
        out = self(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()
        if mask is not None:
            out[~np.asarray(mask, dtype=bool)] = 0.0
        return out.astype(np.float32)


def save_projection_heads(path: str | Path, image_head: ProjectionHead,
                          text_head: ProjectionHead, meta: dict | None = None) -> None:
    """Persist both heads plus metadata (dims, model names) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "image": image_head.state_dict(), "text": text_head.state_dict(),
        "image_in": image_head.in_dim, "text_in": text_head.in_dim,
        "out_dim": image_head.out_dim, "meta": meta or {},
    }, path)


def load_projection_heads(path: str | Path) -> tuple[ProjectionHead, ProjectionHead, dict]:
    """Load heads saved by :func:`save_projection_heads`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"projection weights not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    image_head = ProjectionHead(state["image_in"], state["out_dim"])
    text_head = ProjectionHead(state["text_in"], state["out_dim"])
    image_head.load_state_dict(state["image"])
    text_head.load_state_dict(state["text"])
    return image_head.eval(), text_head.eval(), dict(state.get("meta", {}))


@dataclass
class EncodedBatch:
    """Projected embeddings for a batch of products."""

    image: np.ndarray        # (n, d) projected + normalised; zeros where missing
    text: np.ndarray         # (n, d)
    fused: np.ndarray        # (n, d) or (n, 2d) for concat
    image_mask: np.ndarray   # (n,) bool
    text_mask: np.ndarray    # (n,) bool


class MultimodalEncoder:
    """Image + text encoder with projection heads and fusion.

    Backbones are loaded lazily on first use, so code that only needs the
    projection/fusion logic (e.g. evaluation from cached embeddings) never
    pays the model loading cost, and each process loads each model once.

    Args:
        config: Project configuration.
        image_backbone: Optional pre-built image backbone (used by tests).
        text_backbone: Optional pre-built text backbone (used by tests).
        heads: Optional ``(image_head, text_head)``; otherwise loaded from
            ``config.model.projection_weights`` or initialised from the seed.
    """

    def __init__(self, config: Config, image_backbone: Backbone | None = None,
                 text_backbone: Backbone | None = None,
                 heads: tuple[ProjectionHead, ProjectionHead] | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._image_backbone = image_backbone
        self._text_backbone = text_backbone
        self._heads = heads

    # ------------------------------------------------------------------ backbones
    @property
    def image_backbone(self) -> Backbone:
        if self._image_backbone is None:
            from src.models.image_encoder import CLIPImageEncoder
            self._image_backbone = CLIPImageEncoder(
                self.config.model.image_model, self.device, self.config.model.image_batch_size)
        return self._image_backbone

    @property
    def text_backbone(self) -> Backbone:
        if self._text_backbone is None:
            from src.models.text_encoder import SentenceTextEncoder
            self._text_backbone = SentenceTextEncoder(
                self.config.model.text_model, self.device, self.config.model.text_batch_size)
        return self._text_backbone

    # ------------------------------------------------------------------ heads
    def init_heads(self, image_in: int, text_in: int) -> tuple[ProjectionHead, ProjectionHead]:
        """Create or load projection heads for the given backbone dimensions."""
        m = self.config.model
        if m.projection_weights:
            image_head, text_head, _ = load_projection_heads(self.config.resolve(m.projection_weights))
            logger.info("loaded projection heads from %s", m.projection_weights)
        else:
            image_head = ProjectionHead(image_in, m.projection_dim, m.projection_init, self.config.seed)
            text_head = ProjectionHead(text_in, m.projection_dim, m.projection_init, self.config.seed + 1)
        if (image_head.in_dim, text_head.in_dim) != (image_in, text_in):
            raise ValueError(f"projection heads expect inputs ({image_head.in_dim}, {text_head.in_dim}) "
                             f"but backbones produce ({image_in}, {text_in})")
        self._heads = (image_head.eval(), text_head.eval())
        return self._heads

    @property
    def heads(self) -> tuple[ProjectionHead, ProjectionHead]:
        if self._heads is None:
            self.init_heads(self.image_backbone.dim, self.text_backbone.dim)
        assert self._heads is not None
        return self._heads

    def set_heads(self, image_head: ProjectionHead, text_head: ProjectionHead) -> None:
        """Replace the projection heads (e.g. with the ones used to build an index)."""
        self._heads = (image_head.eval(), text_head.eval())

    # ------------------------------------------------------------------ encoding
    def encode_images_raw(self, images: Sequence[Image.Image | None]) -> tuple[np.ndarray, np.ndarray]:
        """Backbone image embeddings (before projection)."""
        return self.image_backbone.encode(images)

    def encode_texts_raw(self, texts: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """Backbone text embeddings (before projection)."""
        return self.text_backbone.encode(texts)

    def project(self, image_raw: np.ndarray, text_raw: np.ndarray,
                image_mask: np.ndarray, text_mask: np.ndarray) -> EncodedBatch:
        """Project raw backbone embeddings and fuse them."""
        image_head, text_head = self.heads
        image = image_head.project(image_raw, image_mask)
        text = text_head.project(text_raw, text_mask)
        f = self.config.fusion
        fused = fuse(image, text, f.method, f.image_weight, f.text_weight, image_mask, text_mask)
        return EncodedBatch(image, text, fused, np.asarray(image_mask, bool), np.asarray(text_mask, bool))

    def encode(self, images: Sequence[Image.Image | None] | None = None,
               texts: Sequence[str | None] | None = None) -> EncodedBatch:
        """Encode aligned lists of images and titles (either may be omitted).

        Only the needed backbone runs: text-only queries never load CLIP and
        vice versa.
        """
        n = len(images) if images is not None else len(texts or [])
        image_head, text_head = self.heads if self._heads is not None else (None, None)
        if images is not None and any(i is not None for i in images):
            img_raw, img_mask = self.encode_images_raw(images)
        else:
            dim = image_head.in_dim if image_head else self.image_backbone.dim
            img_raw, img_mask = np.zeros((n, dim), np.float32), np.zeros(n, bool)
        if texts is not None and any(t and str(t).strip() for t in texts):
            txt_raw, txt_mask = self.encode_texts_raw(texts)
        else:
            dim = text_head.in_dim if text_head else self.text_backbone.dim
            txt_raw, txt_mask = np.zeros((n, dim), np.float32), np.zeros(n, bool)
        return self.project(img_raw, txt_raw, img_mask, txt_mask)
