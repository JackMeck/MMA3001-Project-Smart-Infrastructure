"""Compatibility entry point for the original project logic.

The legacy implementation is currently stored in the repository's archive and the
notebook conversion. This module provides a proper package import while preserving
those original files for reference.
"""

from __future__ import annotations

# The original notebook-derived code is large and intentionally left in the archive
# for historical reference. This module keeps the package importable and provides a
# clean place for future refactoring and modularisation.

__all__ = [
    "__doc__",
]
