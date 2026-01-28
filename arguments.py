"""Compatibility shim for legacy imports.

New code should import from ``scripts.arguments`` instead of the repo root.
"""

from scripts.arguments import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
