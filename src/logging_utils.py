"""Logging configuration for the asset classification tool.

All modules obtain their logger via::

    import logging
    logger = logging.getLogger(__name__)

The CLI calls configure_logging() once at startup.
Library users configure logging themselves; if they don't, the
NullHandler default silences all output (standard library practice).
"""

from __future__ import annotations

import logging
import sys


def configure_logging(verbosity: int = 0) -> None:
    """Set up root-level logging for CLI use.

    Parameters
    ----------
    verbosity:
        0  → WARNING and above (--quiet)
        1  → INFO and above   (default)
        2  → DEBUG and above  (--verbose)
    """
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = level_map.get(verbosity, logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers (e.g. from previous configure_logging calls in tests)
    root.handlers.clear()
    root.addHandler(handler)


# Silence library output by default — callers opt in by configuring logging.
logging.getLogger(__name__.rsplit(".", 1)[0]).addHandler(logging.NullHandler())
