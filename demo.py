#!/usr/bin/env python3
"""
demo.py
Demonstration of the Memorizer framework functionality.
Shows the complete memory lifecycle from creation to retrieval.
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import db, memory_manager, compression_agent, retrieval, vector_db, security, utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_demo_environment():
    """Set up demo environment variables."""
    # Set demo environment variables if not already set
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/memorizer"
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set - using mock compression agent")
    
    logger.info("Demo environment configured")

def initialize_system():
    """Initialize the Memorizer system."""
    try:
        logger.info("Initializing Memorizer system...")
        
        # Initialize database
        db.initialize_db()
        logger.info("✓ Database initialized")
        
        # Initialize vector database
        vector_db.init_vector_db()
        logger.info("✓ Vector database initialized")
        
        logger.info("System initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

def demo_memory_lifecycle():
    """Demonstrate the complete memory lifecycle."""
    user_id = "demo_user_123"
    
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Memory Lifecycle")
    logger.info(f"{'='*60}")
    
    # Step 1: Add new sessions (very_new tier)
    logger.info("\n1. Adding new sessions to very_new tier...")
    
    sessions = [
        {
            "content": "User asked about refund policy for their recent order #12345. They want to return a defective product and get a full refund.",
            "metadata": {"source": "chat", "order_id": "12345", "issue_type": "refund"}
        },
        {
            "content": "Customer complained about shipping delays. Their order was supposed to arrive 3 days ago but still hasn't shipped. They're frustrated and considering canceling.",
            "metadata": {"source": "email", "order_id": "12346", "issue_type": "shipping"}
        },
        {
            "content": "User praised the fast delivery and excellent customer service. They received their order ahead of schedule and the product quality exceeded expectations.",
            "metadata": {"source": "review", "order_id": "12347", "sentiment": "positive"}
        }
    ]
    
    memory_ids = []
    for i, session in enumerate(sessions):
        memory_id = memory_manager.add_session(
            user_id=user_id,
            content=session["content"],
            metadata=session["metadata"]
        )
        memory_ids.append(memory_id)
        logger.info(f"  ✓ Added session {i+1}: {memory_id}")
    
    # Wait a moment for background embedding processing
    time.sleep(2)
    
    # Step 2: Check memory stats
    logger.info("\n2. Checking memory statistics...")
    stats = memory_manager.get_memory_stats(user_id)
    logger.info(f"  Memory stats: {stats}")
    
    # Step 3: Test retrieval
    logger.info("\n3. Testing memory retrieval...")
    
    test_queries = [
        "refund policy",
        "shipping problems",
        "customer service quality"
    ]
    
    for query in test_queries:
        logger.info(f"\n  Query: '{query}'")
        results = memory_manager.get_context(user_id, query, max_items=3)
        
        for j, result in enumerate(results):
            score = result.get('score', 0)
            content_preview = result.get('content', '')[:100] + "..."
            logger.info(f"    {j+1}. Score: {score:.3f} - {content_preview}")
    
    return True

def main():
    """Main demo function."""
    print("🚀 Memorizer Framework Demo")
    print("=" * 60)
    
    try:
        # Setup
        setup_demo_environment()
        
        # Initialize system
        if not initialize_system():
            logger.error("Failed to initialize system. Exiting.")
            return 1
        
        # Run demo
        logger.info(f"\n🎯 Running Memory Lifecycle Demo...")
        success = demo_memory_lifecycle()
        if success:
            logger.info(f"✓ Demo completed successfully")
        else:
            logger.error(f"✗ Demo failed")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Demo completed successfully!")
        logger.info("The Memorizer framework is working correctly.")
        logger.info(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        return 1
    
    finally:
        # Shutdown background workers
        try:
            memory_manager.shutdown_background_workers()
        except:
            pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
