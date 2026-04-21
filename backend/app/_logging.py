"""Logging helpers that work even when uvicorn replaces the root logger."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "pii") -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(getattr(h, "_pii_handler", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT))
        h._pii_handler = True  # type: ignore[attr-defined]
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
