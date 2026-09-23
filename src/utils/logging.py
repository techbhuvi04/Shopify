"""Project-wide logging setup."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format.

    The log level defaults to INFO and can be changed with the ``LOG_LEVEL``
    environment variable (e.g. ``LOG_LEVEL=DEBUG``).
    """
    global _configured
    if not _configured:
        logging.basicConfig(format=_FORMAT, datefmt="%H:%M:%S",
                            level=os.environ.get("LOG_LEVEL", "INFO").upper())
        # Third-party libraries are noisy at INFO.
        for noisy in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(name)
