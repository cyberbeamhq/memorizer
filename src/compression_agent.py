"""
compression_agent.py
Uses an LLM (default: OpenAI gpt-4o-mini) to compress or summarize content.
Includes safe parsing and retry logic.
"""

import openai

# Compress content into concise summary
def compress_to_mid_term(content: str) -> dict:
    pass

# Aggregate content into long-term insights (<1000 chars, bullet points, sentiment, stats)
def compress_to_long_term(content_list: list[str]) -> dict:
    pass

