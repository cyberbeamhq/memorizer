#!/usr/bin/env python3
"""
agent_memory_example.py
Comprehensive example showing how to use Memorizer for AI agent memory management.
Demonstrates different agent types, memory templates, and integrations.
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import (
    memory_manager, 
    db, 
    vector_db, 
    embeddings,
    agent_interface,
    agent_integrations,
    memory_templates,
    agent_profiles
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for the example."""
    # Set demo environment variables
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/memorizer"
    
    if not os.getenv("EMBEDDING_PROVIDER"):
        os.environ["EMBEDDING_PROVIDER"] = "mock"  # Use mock for demo
    
    logger.info("Environment configured for demo")

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
        
        # Initialize embedding manager
        embeddings.initialize_embedding_manager(
            embeddings.EmbeddingConfig(
                provider=os.getenv("EMBEDDING_PROVIDER", "mock"),
                model="text-embedding-3-small"
            )
        )
        logger.info("✓ Embedding manager initialized")
        
        # Initialize template manager
        memory_templates.initialize_template_manager()
        logger.info("✓ Memory template manager initialized")
        
        # Initialize profile manager
        agent_profiles.initialize_profile_manager()
        logger.info("✓ Agent profile manager initialized")
        
        # Initialize agent memory manager
        agent_interface.initialize_agent_memory_manager(memory_manager)
        logger.info("✓ Agent memory manager initialized")
        
        # Initialize integration manager
        agent_integrations.initialize_integration_manager(memory_manager)
        logger.info("✓ Agent integration manager initialized")
        
        logger.info("System initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False

def demo_conversational_agent():
    """Demonstrate conversational agent memory management."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Conversational Agent Memory Management")
    logger.info(f"{'='*60}")
    
    # Create agent configuration
    agent_config = agent_interface.create_agent_config(
        agent_id="chatbot_001",
        agent_type="conversational",
        framework="general"
    )
    
    # Register agent
    agent_memory_manager = agent_interface.get_agent_memory_manager()
    agent = agent_memory_manager.register_agent(
        agent_interface.AgentConfig(**agent_config)
    )
    
    # Store conversation memories using templates
    template_manager = memory_templates.get_template_manager()
    
    conversations = [
        {
            "user_message": "Hi, I'm looking for a laptop for programming. Budget around $1500.",
            "agent_response": "I'd recommend our Developer Pro Laptop - $1,299, perfect for programming with 16GB RAM and SSD storage. Would you like to see the specs?",
            "session_id": "session_001",
            "conversation_turn": 1,
            "sentiment": "neutral",
            "intent": "product_inquiry"
        },
        {
            "user_message": "I saw that laptop, but I'm also considering the Gaming Beast. Which is better for coding?",
            "agent_response": "Based on your programming needs and $1500 budget, I'd still recommend the Developer Pro. The Gaming Beast is great for gaming but has a louder fan that might be distracting during coding sessions.",
            "session_id": "session_001",
            "conversation_turn": 2,
            "sentiment": "neutral",
            "intent": "product_comparison"
        },
        {
            "user_message": "I'll take the Developer Pro. Can you help me with the checkout?",
            "agent_response": "Perfect choice! The Developer Pro is ideal for your programming needs. Let me help you with checkout. I'll apply a 5% student discount since you mentioned you're a developer.",
            "session_id": "session_001",
            "conversation_turn": 3,
            "sentiment": "positive",
            "intent": "purchase"
        }
    ]
    
    for i, conv in enumerate(conversations):
        # Create memory using conversation template
        memory_data = template_manager.create_memory_from_template(
            "conversation", 
            conv,
            custom_metadata={"demo": "conversational_agent"}
        )
        
        # Store memory
        memory_request = agent_interface.create_memory_request(
            content=memory_data["content"],
            memory_type="conversation",
            session_id=conv["session_id"],
            metadata=memory_data["metadata"],
            tags=["laptop", "programming", "purchase"]
        )
        
        response = agent.store_memory(memory_request)
        logger.info(f"  ✓ Stored conversation {i+1}: {response.memory_id}")
    
    # Retrieve relevant memories
    retrieval_request = agent_interface.create_retrieval_request(
        query="laptop programming budget",
        max_memories=5,
        memory_types=["conversation"]
    )
    
    retrieval_response = agent.retrieve_memories(retrieval_request)
    logger.info(f"  ✓ Retrieved {retrieval_response.total_found} relevant memories")
    
    for i, memory in enumerate(retrieval_response.memories):
        content_preview = memory.get('content', '')[:100] + "..."
        score = memory.get('relevance_score', 0)
        logger.info(f"    {i+1}. Score: {score:.3f} - {content_preview}")
    
    # Get agent stats
    stats = agent.get_memory_stats()
    logger.info(f"  ✓ Agent stats: {stats['total_memories']} total memories")
    
    return True

def demo_task_oriented_agent():
    """Demonstrate task-oriented agent memory management."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Task-Oriented Agent Memory Management")
    logger.info(f"{'='*60}")
    
    # Create agent configuration
    agent_config = agent_interface.create_agent_config(
        agent_id="task_agent_001",
        agent_type="task_oriented",
        framework="general"
    )
    
    # Register agent
    agent_memory_manager = agent_interface.get_agent_memory_manager()
    agent = agent_memory_manager.register_agent(
        agent_interface.AgentConfig(**agent_config)
    )
    
    # Store task execution memories
    template_manager = memory_templates.get_template_manager()
    
    tasks = [
        {
            "task_description": "Process customer order #12345 and update inventory",
            "execution_status": "completed",
            "task_id": "task_001",
            "execution_time": 45,
            "tools_used": ["inventory_api", "order_processor"],
            "output": "Order processed successfully, inventory updated"
        },
        {
            "task_description": "Generate monthly sales report for Q1 2024",
            "execution_status": "completed",
            "task_id": "task_002",
            "execution_time": 120,
            "tools_used": ["data_analyzer", "report_generator"],
            "output": "Report generated with 15% increase in sales"
        },
        {
            "task_description": "Backup database to cloud storage",
            "execution_status": "failed",
            "task_id": "task_003",
            "execution_time": 30,
            "tools_used": ["backup_tool"],
            "errors": "Connection timeout to cloud storage"
        }
    ]
    
    for i, task in enumerate(tasks):
        # Create memory using task execution template
        memory_data = template_manager.create_memory_from_template(
            "task_execution", 
            task,
            custom_metadata={"demo": "task_oriented_agent"}
        )
        
        # Store memory
        memory_request = agent_interface.create_memory_request(
            content=memory_data["content"],
            memory_type="task_execution",
            metadata=memory_data["metadata"],
            priority=3 if task["execution_status"] == "failed" else 2,
            tags=["automation", "business_process"]
        )
        
        response = agent.store_memory(memory_request)
        logger.info(f"  ✓ Stored task {i+1}: {response.memory_id}")
    
    # Retrieve task-related memories
    retrieval_request = agent_interface.create_retrieval_request(
        query="failed tasks backup database",
        max_memories=5,
        memory_types=["task_execution"]
    )
    
    retrieval_response = agent.retrieve_memories(retrieval_request)
    logger.info(f"  ✓ Retrieved {retrieval_response.total_found} relevant task memories")
    
    # Get agent stats
    stats = agent.get_memory_stats()
    logger.info(f"  ✓ Agent stats: {stats['total_memories']} total memories")
    
    return True

def demo_analytical_agent():
    """Demonstrate analytical agent memory management."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Analytical Agent Memory Management")
    logger.info(f"{'='*60}")
    
    # Create agent configuration
    agent_config = agent_interface.create_agent_config(
        agent_id="analyst_001",
        agent_type="analytical",
        framework="general"
    )
    
    # Register agent
    agent_memory_manager = agent_interface.get_agent_memory_manager()
    agent = agent_memory_manager.register_agent(
        agent_interface.AgentConfig(**agent_config)
    )
    
    # Store analytical memories
    template_manager = memory_templates.get_template_manager()
    
    analyses = [
        {
            "decision_context": "Analyzing customer churn patterns to improve retention",
            "decision_made": "Implement proactive customer outreach program",
            "reasoning": "Data shows 80% of churned customers had no contact in 30 days before churning. Proactive outreach can reduce churn by 25%.",
            "alternatives_considered": ["Price reduction", "Feature improvements", "Better onboarding"],
            "confidence_score": 0.85,
            "decision_factors": ["historical_data", "customer_behavior", "cost_analysis"]
        },
        {
            "learning_topic": "Market trend analysis",
            "knowledge_gained": "AI adoption in healthcare is growing 40% annually, with telemedicine and diagnostic tools leading the trend. Key drivers include cost reduction and improved patient outcomes.",
            "source": "industry_report_2024",
            "confidence_level": 0.9,
            "application_context": "Healthcare AI strategy planning",
            "related_concepts": ["telemedicine", "diagnostic_ai", "healthcare_automation"]
        }
    ]
    
    for i, analysis in enumerate(analyses):
        if "decision_context" in analysis:
            # Decision making memory
            memory_data = template_manager.create_memory_from_template(
                "decision_making", 
                analysis,
                custom_metadata={"demo": "analytical_agent"}
            )
            memory_type = "decision"
        else:
            # Learning memory
            memory_data = template_manager.create_memory_from_template(
                "learning", 
                analysis,
                custom_metadata={"demo": "analytical_agent"}
            )
            memory_type = "learning"
        
        # Store memory
        memory_request = agent_interface.create_memory_request(
            content=memory_data["content"],
            memory_type=memory_type,
            metadata=memory_data["metadata"],
            priority=3,
            tags=["analysis", "strategy", "insights"]
        )
        
        response = agent.store_memory(memory_request)
        logger.info(f"  ✓ Stored analysis {i+1}: {response.memory_id}")
    
    # Retrieve analytical insights
    retrieval_request = agent_interface.create_retrieval_request(
        query="customer churn retention strategy",
        max_memories=5,
        memory_types=["decision", "learning"]
    )
    
    retrieval_response = agent.retrieve_memories(retrieval_request)
    logger.info(f"  ✓ Retrieved {retrieval_response.total_found} relevant analytical memories")
    
    # Get agent stats
    stats = agent.get_memory_stats()
    logger.info(f"  ✓ Agent stats: {stats['total_memories']} total memories")
    
    return True

def demo_agent_integrations():
    """Demonstrate agent framework integrations."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Agent Framework Integrations")
    logger.info(f"{'='*60}")
    
    # Get integration manager
    integration_manager = agent_integrations.get_integration_manager()
    
    # List available frameworks
    frameworks = integration_manager.list_available_frameworks()
    logger.info(f"  Available frameworks: {frameworks}")
    
    # Demo LangChain integration (if enabled)
    if "langchain" in frameworks:
        logger.info("  ✓ LangChain integration available")
        
        # Create agent memory
        agent_memory = agent_integrations.AgentMemory(
            agent_id="langchain_agent_001",
            session_id="session_001",
            content="User asked about weather in New York",
            metadata={"intent": "weather_inquiry", "location": "New York"},
            timestamp=datetime.now(),
            memory_type="conversation"
        )
        
        # Store memory
        memory_id = integration_manager.store_agent_memory("langchain", agent_memory)
        logger.info(f"  ✓ Stored LangChain memory: {memory_id}")
        
        # Retrieve memories
        context = agent_integrations.AgentContext(
            agent_id="langchain_agent_001",
            query="weather New York"
        )
        
        memories = integration_manager.retrieve_agent_memories("langchain", context)
        logger.info(f"  ✓ Retrieved {len(memories)} LangChain memories")
    
    # Demo other integrations
    for framework in ["llamaindex", "autogpt", "crewai"]:
        if framework in frameworks:
            logger.info(f"  ✓ {framework.title()} integration available")
    
    return True

def demo_memory_lifecycle():
    """Demonstrate memory lifecycle management."""
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: Memory Lifecycle Management")
    logger.info(f"{'='*60}")
    
    # Create a test agent
    agent_config = agent_interface.create_agent_config(
        agent_id="lifecycle_test_001",
        agent_type="general",
        framework="general"
    )
    
    agent_memory_manager = agent_interface.get_agent_memory_manager()
    agent = agent_memory_manager.register_agent(
        agent_interface.AgentConfig(**agent_config)
    )
    
    # Add multiple memories to trigger lifecycle
    for i in range(25):  # More than very_new_limit
        memory_request = agent_interface.create_memory_request(
            content=f"Test memory {i+1} for lifecycle demonstration",
            memory_type="conversation",
            metadata={"test": True, "iteration": i+1}
        )
        
        response = agent.store_memory(memory_request)
        logger.info(f"  ✓ Stored memory {i+1}: {response.memory_id}")
    
    # Check memory stats
    stats = agent.get_memory_stats()
    logger.info(f"  Memory stats: {stats['memory_stats']}")
    
    # Simulate memory lifecycle movement
    logger.info("  Simulating memory lifecycle movement...")
    moved = memory_manager.move_memory_between_tiers(
        user_id=agent.user_id,
        very_new_limit=20,
        mid_term_limit=200
    )
    logger.info(f"  ✓ Moved memories: {moved}")
    
    # Check stats after movement
    stats_after = agent.get_memory_stats()
    logger.info(f"  Memory stats after movement: {stats_after['memory_stats']}")
    
    return True

def main():
    """Main demo function."""
    print("🚀 Memorizer Framework - AI Agent Memory Management Demo")
    print("=" * 80)
    
    try:
        # Setup
        setup_environment()
        
        # Initialize system
        if not initialize_system():
            logger.error("Failed to initialize system. Exiting.")
            return 1
        
        # Run demos
        demos = [
            ("Conversational Agent", demo_conversational_agent),
            ("Task-Oriented Agent", demo_task_oriented_agent),
            ("Analytical Agent", demo_analytical_agent),
            ("Agent Integrations", demo_agent_integrations),
            ("Memory Lifecycle", demo_memory_lifecycle)
        ]
        
        for demo_name, demo_func in demos:
            logger.info(f"\n🎯 Running {demo_name} Demo...")
            try:
                success = demo_func()
                if success:
                    logger.info(f"✓ {demo_name} demo completed successfully")
                else:
                    logger.error(f"✗ {demo_name} demo failed")
            except Exception as e:
                logger.error(f"✗ {demo_name} demo failed with error: {e}")
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("🎉 All demos completed!")
        logger.info("The Memorizer framework is ready for AI agent memory management.")
        logger.info(f"{'='*80}")
        
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
