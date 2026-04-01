from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: str = "app.log",
    force: bool = False,
) -> None:
    """
    Configure root logging once for the whole app.
    Avoids per-module FileHandler duplication.
    """
    root = logging.getLogger()
    if root.handlers and not force:
        return

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=force)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)