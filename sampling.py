"""Compatibility shim for sampling helpers.

The canonical module lives in :mod:`data.processing.sampling`.
"""

from data.processing.sampling import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
