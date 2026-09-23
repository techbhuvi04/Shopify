"""Sentence-Transformer title encoder.

Why a Sentence Transformer instead of TF-IDF? Seller titles for the same
product use different word order, abbreviations and synonyms ("tee" vs
"t-shirt", "flask" vs "bottle"). Transformer sentence embeddings are trained
for semantic similarity and handle these paraphrases, while still being cheap
enough (MiniLM: 22M parameters) to run on a laptop CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTextEncoder:
    """Batch text encoder backed by ``sentence-transformers``.

    Args:
        model_name: Sentence-Transformers model ID.
        device: Torch device to run on.
        batch_size: Titles per forward pass.
    """

    def __init__(self, model_name: str, device: torch.device | str = "cpu",
                 batch_size: int = 64) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        logger.info("loading text model %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=str(self.device))
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim: int = int(get_dim())

    def encode(self, texts: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
        """Encode titles into L2-normalised sentence embeddings.

        Args:
            texts: Titles; ``None`` or blank strings mark missing titles.

        Returns:
            ``(embeddings, mask)`` with zero rows where the title was empty.
        """
        n = len(texts)
        emb = np.zeros((n, self.dim), dtype=np.float32)
        mask = np.array([bool(t and str(t).strip()) for t in texts], dtype=bool)
        valid = np.flatnonzero(mask)
        if len(valid):
            emb[valid] = self.model.encode(
                [str(texts[i]) for i in valid], batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float32)
        return emb, mask
