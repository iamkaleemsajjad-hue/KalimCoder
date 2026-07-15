"""
logging.py — Centralized logging setup for KaleemCoder.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Shared format used by every pipeline script
_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger.

    Args:
        name:     Logger name (usually __name__ of calling module).
        log_file: Optional path to write logs to disk.
        level:    Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(level)
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def configure_pipeline_logging(
    log_dir: Path,
    log_prefix: str,
    logger_name: str,
    verbose: bool = False,
) -> logging.Logger:
    """Configure the root logger for a pipeline script.

    Sets up:
    * A rotating **file handler** (always DEBUG) under *log_dir*.
    * A **console handler** (INFO, or DEBUG when *verbose*).

    All pipeline scripts should call this once at startup instead of
    maintaining their own ``_configure_logging`` copies.

    Parameters
    ----------
    log_dir:
        Directory where the timestamped log file is written.
        Created automatically if it does not exist.
    log_prefix:
        Prefix for the log filename, e.g. ``"download"`` →
        ``logs/download/download_20240715_120000.log``.
    logger_name:
        Name passed to :func:`logging.getLogger` for the returned logger.
    verbose:
        When ``True`` the console handler emits DEBUG messages.

    Returns
    -------
    logging.Logger
        A ready-to-use logger named *logger_name*.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger(logger_name)
