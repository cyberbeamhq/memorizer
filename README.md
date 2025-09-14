# Memorizer

[![Build](https://img.shields.io/github/actions/workflow/status/your-org/memorizer/ci.yml?branch=main)](https://github.com/your-org/memorizer/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20gpt--4o--mini-green)](https://platform.openai.com/)

**Memorizer** is an open-source memory lifecycle framework for AI assistants and agents.  
It helps LLM-based systems **forget smartly, remember what matters, and reduce token usage** while keeping historical context accessible.

---

## ✨ Key Features

- **Three-tier memory lifecycle**
  - **Very new** → recent sessions (raw, full text, up to 10 days / N sessions).
  - **Mid-term** → compressed summaries with unnecessary words removed (last 12 months).
  - **Long-term** → highly aggregated, <1000-character briefs with sentiment, preferences, and key metrics.
- **DB-first design (Postgres + JSONB)**  
  Structured queries and lifecycle management with SQL, cheap storage, and easy analytics.
- **Compression Agent**  
  LLM-powered summarizer (default: OpenAI `gpt-4o-mini`) with safe parsing, retries, and fallback.
- **Hybrid retrieval**  
  Cheap keyword relevance scoring first, with optional vector DB fallback (Pinecone, Weaviate, Chroma, PostgreSQL `pgvector`).
- **Production-ready infrastructure**  
  Works with AWS, Azure, or local deployments. Integrates easily with **LangChain**, **LlamaIndex**, or standalone agents.
- **Pluggable architecture**  
  Swap out vector DBs, LLM providers, or customize compression rules.

---

## 🏗️ Architecture

```
┌───────────────────────┐
│     Very New Memory   │ (raw sessions: up to 10 days / N sessions)
└─────────┬─────────────┘
          │ move/compress
          ▼
┌───────────────────────┐
│    Mid-Term Memory    │ (summaries of last 12 months, trimmed text)
└─────────┬─────────────┘
          │ aggregate
          ▼
┌───────────────────────┐
│   Long-Term Memory    │ (aggregated briefs <1k chars, stats, prefs)
└─────────┬─────────────┘
          │ fallback for deep recall
          ▼
┌───────────────────────┐
│   Vector DB (optional)│ (Pinecone / Weaviate / Chroma / pgvector)
└───────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/your-org/memorizer.git
cd memorizer
```

### 2. Set up environment
Copy the example .env file and update it with your credentials:

```bash
cp .env.example .env
```

You'll need:
- `OPENAI_API_KEY` (or other LLM provider key)
- `DATABASE_URL` (Postgres connection string)
- Optional vector DB API keys (e.g. Pinecone)

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run database migrations
```bash
python scripts/init_db.py
```

### 5. Try a local demo
```bash
python demo.py
```

---

## 🔧 Example: E-commerce AI Assistant

Memorizer can back an e-commerce assistant:

- **Very new memory**: Last 5 support chats ("Where is my order?")
- **Mid-term memory**: Summarized chat history ("Customer had 12 refund requests in 2024")
- **Long-term memory**: Aggregated insights ("Customer prefers express shipping, positive sentiment about product quality, negative about delivery speed")

When the customer chats again:
1. Assistant retrieves relevant context from Memorizer.
2. Uses hybrid retrieval: keyword search for "refund", vector fallback for older "delivery delay" issues.
3. Responds with awareness of customer history, without blowing up tokens.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+, Postgres (with JSONB)
- **Vector DB options**: Pinecone, Weaviate, Chroma, pgvector
- **LLM compression**: OpenAI (gpt-4o-mini) by default, pluggable with local or third-party LLMs
- **Frameworks**: Ready for integration with LangChain, LlamaIndex, or standalone

---

## 📂 Repository Structure

```
src/
  db.py                # Database schema + queries
  memory_manager.py    # Orchestration of memory lifecycle
  compression_agent.py # Summarization + compression
  retrieval.py         # Context retrieval with hybrid scoring
  vector_db.py         # Abstraction over vector DBs
demo.py                # Example usage
scripts/
  init_db.py           # Create tables, run migrations
.env.example           # Environment variables
requirements.txt
```

---

## 📊 Roadmap

- [ ] LangChain + LlamaIndex adapters
- [ ] Provenance metadata (why was something kept/removed?)
- [ ] Audit logging + RBAC hooks
- [ ] Declarative compression policies (rules before LLM summarization)
- [ ] Monitoring & metrics (memory growth, retrieval latency, token savings)
- [ ] Example apps (E-commerce, CRM, Knowledge Assistant)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR if you'd like to:

- Add vector DB connectors
- Improve compression strategies
- Provide more example use cases

---

## 📜 License

Apache License 2.0.  
See [LICENSE](./LICENSE) for details.
