#!/usr/bin/env python
"""Convenience launcher: ``python scripts/run_pipeline.py --help``."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from occupancy.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
