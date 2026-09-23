"""Title cleaning and robust image loading."""

from __future__ import annotations

import html
import io
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Marketplace boilerplate that carries no product identity.
_PROMO_PATTERNS = [
    r"\[[^\]]*\]",                 # [PROMO], [READY STOCK], ...
    r"\bfree\s+shipping\b",
    r"\bready\s+stock\b",
    r"\bbest\s+seller\b",
    r"\b100%\s*original\b",
    r"-\s*original\b",
]
_PROMO_RE = re.compile("|".join(_PROMO_PATTERNS), flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_title(title: object, remove_promo: bool = True) -> str:
    """Normalise a product title.

    Steps: coerce to string (``None``/NaN become ``""``), unescape HTML
    entities, Unicode NFKC normalisation, optional removal of promo
    boilerplate, lower-casing and whitespace collapsing.

    Args:
        title: Raw title value (may be ``None`` or NaN).
        remove_promo: Strip marketplace promo phrases such as "free shipping".

    Returns:
        The cleaned title; an empty string if nothing usable remains.
    """
    if title is None or (isinstance(title, float) and title != title):  # NaN check
        return ""
    text = unicodedata.normalize("NFKC", html.unescape(str(title)))
    if remove_promo:
        text = _PROMO_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip(" -").lower()
    return text


def load_image(source: str | Path | bytes | Image.Image | None) -> Image.Image | None:
    """Load an image as RGB, returning ``None`` instead of raising on failure.

    Accepts a filesystem path, raw bytes (e.g. an HTTP upload) or an existing
    PIL image. EXIF orientation is applied so phone photos are upright.
    """
    if source is None:
        return None
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, (bytes, bytearray)):
            if not source:
                return None
            img = Image.open(io.BytesIO(source))
        else:
            path = Path(source)
            if not path.is_file():
                logger.debug("image not found: %s", path)
                return None
            img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("could not read image %s: %s", source if not isinstance(source, bytes) else "<bytes>", exc)
        return None
