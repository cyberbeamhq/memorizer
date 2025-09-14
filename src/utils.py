"""
utils.py
Utility functions shared across the framework.
"""

import time
import json

# Retry wrapper for external API calls (e.g., LLM requests)
def retry(fn, retries: int = 3, delay: float = 1.0):
    pass

# Safe JSON parsing
def safe_parse_json(raw: str, fallback: dict = None):
    pass

# Timestamp generator
def now_ts() -> str:
    pass

