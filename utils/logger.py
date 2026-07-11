"""
Centralized logging setup.

Every module calls get_logger(name, logfile) to obtain a logger that writes
to both the console and a dedicated file inside logs/. This keeps
training / evaluation / runtime / camera / model-loading logs separated
and easy to inspect after the fact.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED_LOGGERS: dict[str, logging.Logger] = {}

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    logfile: Optional[str] = None,
    logs_dir: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or fetch a cached) logger.

    Args:
        name: logger name, typically __name__ or a component name
              ("training", "evaluation", "camera", "model_loading", "runtime").
        logfile: filename (relative to logs_dir) to write to, e.g. "training.log".
                 If None, defaults to "<name>.log".
        logs_dir: directory to store log files. Defaults to <project_root>/logs.
        level: logging level.
    """
    cache_key = f"{name}:{logfile}"
    if cache_key in _CONFIGURED_LOGGERS:
        return _CONFIGURED_LOGGERS[cache_key]

    if logs_dir is None:
        # Local import to avoid circular import with config at module load time
        from config.config import config as _cfg
        logs_dir = _cfg.logs_dir

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logfile = logfile or f"{name}.log"
    log_path = logs_dir / logfile

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _CONFIGURED_LOGGERS[cache_key] = logger
    return logger
