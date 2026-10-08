"""Nonfatal diagnostics: concise by default, traceback when DEBUG is enabled.

Library code does not configure global handlers; embedders keep control of logging.
"""
from __future__ import annotations

import logging
import sys


def report_exception(logger: logging.Logger, message: str, *args, level: int = logging.WARNING) -> None:
    """Report the currently handled exception without a default traceback dump."""
    info = sys.exc_info()
    error = info[1]
    detail = ' '.join(str(error).split())
    logger.log(level, message + " (%s: %s)", *args, type(error).__name__, detail,
               exc_info=info if logger.isEnabledFor(logging.DEBUG) else None)
