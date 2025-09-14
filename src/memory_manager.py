"""
memory_manager.py
High-level orchestration of memory lifecycle.
Coordinates between db.py, compression_agent.py, retrieval.py, and vector_db.py.
"""

from . import db, compression_agent, retrieval, vector_db

# Add a new user interaction (goes into very_new)
def add_session(user_id: str, content: str, metadata: dict = None):
    pass

# Periodic job: compress/move old memories into mid_term or long_term
def move_memory_between_tiers(user_id: str):
    pass

# Retrieve context for an active session
def get_context(user_id: str, query: str, max_items: int = 5):
    pass
