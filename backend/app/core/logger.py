"""Logging configuration utilities."""

from __future__ import annotations

import logging
from logging.config import dictConfig


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
        }
    },
    "root": {"handlers": ["default"], "level": "INFO"},
}


def configure_logging() -> None:
    """Configure process-wide logging."""

    dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""

    return logging.getLogger(name)
