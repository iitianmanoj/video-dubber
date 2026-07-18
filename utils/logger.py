"""
utils/logger.py
----------------
Rich-powered logging. Every run writes a timestamped log file to `logs/`
in addition to pretty console output, so failures can be diagnosed after
the fact (processing time per stage, warnings, full tracebacks on error).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rich.logging import RichHandler

from config import LOGS_DIR

_LOG_FORMAT = "%(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once per process and return the app logger."""
    global _configured

    logger = logging.getLogger("video_dubber")

    if _configured:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    console_handler.setLevel(level)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOGS_DIR) / f"run_{timestamp}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    file_handler.setLevel(logging.DEBUG)  # file gets everything

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug("Logging initialized. Writing full log to %s", log_path)
    _configured = True
    return logger


def get_logger() -> logging.Logger:
    """Fetch the app logger, configuring it with defaults if needed."""
    return setup_logging()


class StageTimer:
    """Context manager that logs how long a pipeline stage took.

    Usage:
        with StageTimer("Transcription"):
            do_work()
    """

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.logger = get_logger()
        self.start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = time.perf_counter()
        self.logger.info("[bold cyan]→ %s...[/bold cyan]", self.stage_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            self.logger.info(
                "[bold green]✓ %s completed[/bold green] (%.1fs)",
                self.stage_name, elapsed,
            )
        else:
            self.logger.error(
                "[bold red]✗ %s failed[/bold red] after %.1fs: %s",
                self.stage_name, elapsed, exc_val,
            )
        return False  # never swallow exceptions
