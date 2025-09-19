# 🧠 Memorizer Framework - Usage Guide

A flexible memory management framework designed specifically for AI agents. This guide shows you how to integrate Memorizer into your AI agent applications.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd memorizer

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp env.example .env
# Edit .env with your configuration
```

### 2. Basic Setup

```python
import os
from src import memory_manager, db, vector_db, agent_interface

# Set up environment
os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost:5432/memorizer"
os.environ["EMBEDDING_PROVIDER"] = "openai"  # or "mock" for testing
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Initialize system
db.initialize_db()
vector_db.init_vector_db()

# Initialize agent memory manager
agent_interface.initialize_agent_memory_manager(memory_manager)
```

### 3. Create Your First Agent

```python
from src.agent_interface import AgentConfig, AgentType

# Create agent configuration
config = AgentConfig(
    agent_id="my_chatbot",
    agent_type=AgentType.CONVERSATIONAL,
    framework="general",
    context_window=10,
    max_tokens=4000
)

# Register agent
agent_memory_manager = agent_interface.get_agent_memory_manager()
agent = agent_memory_manager.register_agent(config)
```

## 📝 Memory Management

### Storing Memories

```python
from src.agent_interface import MemoryRequest, MemoryType

# Store a conversation memory
memory_request = MemoryRequest(
    content="User asked about weather in New York",
    memory_type=MemoryType.CONVERSATION,
    session_id="session_001",
    metadata={"intent": "weather_inquiry", "location": "New York"},
    tags=["weather", "location"]
)

response = agent.store_memory(memory_request)
print(f"Memory stored: {response.memory_id}")
```

### Retrieving Memories

```python
from src.agent_interface import RetrievalRequest

# Retrieve relevant memories
retrieval_request = RetrievalRequest(
    query="weather New York",
    max_memories=5,
    memory_types=[MemoryType.CONVERSATION]
)

retrieval_response = agent.retrieve_memories(retrieval_request)
print(f"Found {retrieval_response.total_found} relevant memories")

for memory in retrieval_response.memories:
    print(f"- {memory['content'][:100]}...")
```

## 🎯 Agent Types

### Conversational Agent

Perfect for chatbots and dialogue systems:

```python
# Create conversational agent
config = AgentConfig(
    agent_id="chatbot_001",
    agent_type=AgentType.CONVERSATIONAL,
    context_window=10,
    max_tokens=4000
)

agent = agent_memory_manager.register_agent(config)

# Store conversation
memory_request = MemoryRequest(
    content="User: Hello\nAgent: Hi! How can I help you today?",
    memory_type=MemoryType.CONVERSATION,
    metadata={"sentiment": "positive", "intent": "greeting"}
)

agent.store_memory(memory_request)
```

### Task-Oriented Agent

Ideal for workflow automation and task execution:

```python
# Create task-oriented agent
config = AgentConfig(
    agent_id="task_agent_001",
    agent_type=AgentType.TASK_ORIENTED,
    context_window=8,
    max_tokens=3000
)

agent = agent_memory_manager.register_agent(config)

# Store task execution
memory_request = MemoryRequest(
    content="Task: Process order #12345\nStatus: Completed\nTools: [order_processor, inventory_api]",
    memory_type=MemoryType.TASK_EXECUTION,
    metadata={"task_id": "task_001", "execution_time": 45},
    priority=3
)

agent.store_memory(memory_request)
```

### Analytical Agent

Great for data analysis and research:

```python
# Create analytical agent
config = AgentConfig(
    agent_id="analyst_001",
    agent_type=AgentType.ANALYTICAL,
    context_window=15,
    max_tokens=6000
)

agent = agent_memory_manager.register_agent(config)

# Store analysis
memory_request = MemoryRequest(
    content="Analysis: Customer churn increased 15% in Q1\nDecision: Implement retention program\nReasoning: Based on behavioral patterns",
    memory_type=MemoryType.DECISION,
    metadata={"confidence": 0.85, "data_source": "analytics_dashboard"},
    priority=3
)

agent.store_memory(memory_request)
```

## 🏗️ Memory Templates

Use predefined templates for structured memory storage:

```python
from src import memory_templates

# Get template manager
template_manager = memory_templates.get_template_manager()

# Create conversation memory using template
conversation_data = {
    "user_message": "I need help with my order",
    "agent_response": "I'd be happy to help! What's your order number?",
    "session_id": "session_001",
    "sentiment": "neutral",
    "intent": "customer_support"
}

memory_data = template_manager.create_memory_from_template(
    "conversation", 
    conversation_data
)

# Store using template
memory_request = MemoryRequest(
    content=memory_data["content"],
    memory_type=MemoryType.CONVERSATION,
    metadata=memory_data["metadata"]
)

agent.store_memory(memory_request)
```

Available templates:
- `conversation` - For dialogue interactions
- `task_execution` - For task completion tracking
- `decision_making` - For decision processes
- `error_handling` - For error resolution
- `user_preference` - For user preferences
- `tool_usage` - For tool execution logs
- `goal_tracking` - For goal progress
- `learning` - For knowledge acquisition

## 🔌 Framework Integrations

### LangChain Integration

```python
from src import agent_integrations

# Enable LangChain integration in .env
# LANGCHAIN_ENABLED=true

# Get integration manager
integration_manager = agent_integrations.get_integration_manager()

# Store LangChain agent memory
agent_memory = agent_integrations.AgentMemory(
    agent_id="langchain_agent_001",
    session_id="session_001",
    content="User asked about weather",
    metadata={"intent": "weather_inquiry"},
    timestamp=datetime.now(),
    memory_type="conversation"
)

memory_id = integration_manager.store_agent_memory("langchain", agent_memory)
```

### Other Framework Integrations

```python
# Available integrations:
# - langchain
# - llamaindex  
# - autogpt
# - crewai

# Enable in .env:
# LANGCHAIN_ENABLED=true
# LLAMAINDEX_ENABLED=true
# AUTOGPT_ENABLED=true
# CREWAI_ENABLED=true
```

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/memorizer

# Embedding Provider
EMBEDDING_PROVIDER=openai  # openai, cohere, huggingface, mock
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Memory Lifecycle
VERY_NEW_LIMIT=20
MID_TERM_LIMIT=200
LONG_TERM_LIMIT=1000
VERY_NEW_DAYS=10
MID_TERM_DAYS=365

# Agent Configuration
DEFAULT_AGENT_TYPE=general
DEFAULT_AGENT_CONTEXT_WINDOW=5
DEFAULT_AGENT_MAX_TOKENS=4000
```

### Agent Profiles

Use predefined profiles for different agent types:

```python
from src import agent_profiles

# Get profile manager
profile_manager = agent_profiles.get_profile_manager()

# Get agent profile
profile = profile_manager.get_profile("conversational")
print(f"Profile: {profile.profile_name}")
print(f"Context window: {profile.context_window}")
print(f"Memory limits: {profile.very_new_limit}/{profile.mid_term_limit}/{profile.long_term_limit}")

# Create agent with profile
agent_config = agent_profiles.create_agent_with_profile(
    agent_id="my_agent",
    agent_type="conversational",
    framework="general"
)
```

## 📊 Monitoring and Stats

### Get Agent Statistics

```python
# Get memory statistics
stats = agent.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"By tier: {stats['memory_stats']}")

# Get all agent stats
agent_memory_manager = agent_interface.get_agent_memory_manager()
all_stats = agent_memory_manager.get_all_agent_stats()
for agent_id, agent_stats in all_stats.items():
    print(f"{agent_id}: {agent_stats['total_memories']} memories")
```

### Memory Lifecycle Management

```python
# Move memories between tiers
moved = memory_manager.move_memory_between_tiers(
    user_id=agent.user_id,
    very_new_limit=20,
    mid_term_limit=200
)

print(f"Moved to mid-term: {len(moved['to_mid_term'])}")
print(f"Moved to long-term: {len(moved['to_long_term'])}")

# Clean up old memories
cleaned = memory_manager.cleanup_old_memories(
    user_id=agent.user_id,
    max_age_days=365
)
print(f"Cleaned up {cleaned} old memories")
```

## 🔍 Advanced Usage

### Custom Memory Templates

```python
from src.memory_templates import MemoryTemplate, TemplateType

# Create custom template
custom_template = MemoryTemplate(
    template_type=TemplateType.CUSTOM,
    name="Custom Business Logic",
    description="Template for business-specific memory",
    required_fields=["business_context", "action_taken", "outcome"],
    optional_fields=["stakeholders", "timeline", "impact"],
    default_metadata={"memory_type": "business_logic", "priority": 2}
)

# Add to template manager
template_manager = memory_templates.get_template_manager()
template_manager.add_custom_template(custom_template)

# Use custom template
business_data = {
    "business_context": "Customer complaint about delayed shipment",
    "action_taken": "Expedited shipping and provided discount",
    "outcome": "Customer satisfied, issue resolved",
    "stakeholders": ["customer", "logistics_team"],
    "impact": "positive"
}

memory_data = template_manager.create_memory_from_template(
    "custom_business_logic", 
    business_data
)
```

### Custom Agent Profiles

```python
from src.agent_profiles import MemoryProfile, CompressionStrategy, RetrievalStrategy

# Create custom profile
custom_profile = MemoryProfile(
    profile_name="Custom Business Agent",
    agent_type="business_automation",
    description="Profile for business process automation",
    very_new_limit=30,
    mid_term_limit=300,
    long_term_limit=1500,
    very_new_days=7,
    mid_term_days=90,
    long_term_days=365,
    compression_strategy=CompressionStrategy.BALANCED,
    context_window=12,
    max_retrieval_items=15,
    priority_memory_types=["task_execution", "decision", "business_logic"],
    custom_settings={
        "business_rules": True,
        "compliance_tracking": True,
        "audit_logging": True
    }
)

# Add to profile manager
profile_manager = agent_profiles.get_profile_manager()
profile_manager.create_custom_profile(custom_profile)
```

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t memorizer .

# Run with Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
```

### Health Checks

```python
from src import health

# Check system health
health_checker = health.HealthChecker()
status = health_checker.check_all()

if status["overall"]["healthy"]:
    print("System is healthy")
else:
    print(f"System issues: {status['issues']}")
```

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `agent_memory_example.py` - Complete demonstration of all features
- `demo.py` - Basic framework demonstration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the documentation
- Look at examples
- Open an issue for bugs
- Start a discussion for questions

---

**Happy coding with Memorizer! 🧠✨**
