#!/usr/bin/env python3
"""Compatibility entry point; use ``generate_tiny.py`` for any TP/EP layout."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("generate_tiny.py")), run_name="__main__")
