#!/usr/bin/env python3
"""
init_db.py
Initialize the Memorizer database schema.
"""
import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database."""
    print("🗄️  Initializing Memorizer Database")
    print("=" * 50)
    
    try:
        # Set default database URL if not provided
        if not os.getenv("DATABASE_URL"):
            os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/memorizer"
            logger.info("Using default database URL")
        
        # Initialize database
        db.initialize_db()
        
        print("✅ Database initialized successfully!")
        print("\nNext steps:")
        print("1. Run the demo: python demo.py")
        print("2. Start using the Memorizer framework in your application")
        
        return 0
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        print(f"❌ Database initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your DATABASE_URL environment variable")
        print("3. Ensure the database exists and you have proper permissions")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
