"""CLIP image encoder.

Why CLIP? CLIP's vision tower was trained contrastively on ~400M image-text
pairs, so its embedding space groups images by *semantic* content (object
type, colour, style) rather than raw pixels. That makes it a strong zero-shot
backbone for "is this the same product?" without any task-specific training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CLIPImageEncoder:
    """Batch image encoder backed by a Hugging Face CLIP model.

    Args:
        model_name: Hugging Face model ID of a CLIP-family checkpoint.
        device: Torch device to run on.
        batch_size: Images per forward pass.
    """

    def __init__(self, model_name: str, device: torch.device | str = "cpu",
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        logger.info("loading image model %s on %s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.dim: int = int(self.model.config.projection_dim)

    @torch.inference_mode()
    def _forward(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model.get_image_features(pixel_values=pixel_values)
        # transformers>=5 returns a model output; older versions return a tensor.
        feats = out if isinstance(out, torch.Tensor) else out.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    def encode(self, images: Sequence[Image.Image | None]) -> tuple[np.ndarray, np.ndarray]:
        """Encode images into L2-normalised CLIP embeddings.

        Args:
            images: PIL images; ``None`` entries mark missing images.

        Returns:
            ``(embeddings, mask)`` where ``embeddings`` is ``(n, dim)`` float32
            with zero rows for missing images and ``mask`` is a boolean array
            that is ``True`` where an image was encoded.
        """
        n = len(images)
        emb = np.zeros((n, self.dim), dtype=np.float32)
        mask = np.array([img is not None for img in images], dtype=bool)
        valid = np.flatnonzero(mask)
        for start in range(0, len(valid), self.batch_size):
            rows = valid[start:start + self.batch_size]
            emb[rows] = self._forward([images[i] for i in rows]).cpu().numpy()
        return emb, mask
