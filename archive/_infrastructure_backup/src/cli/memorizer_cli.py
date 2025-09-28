#!/usr/bin/env python3
"""
memorizer_cli.py
Command-line interface for the Memorizer Framework.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ..core.cli.main import main

if __name__ == "__main__":
    main()
