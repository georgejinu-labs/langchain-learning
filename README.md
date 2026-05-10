# LangChain Learning

A hands-on project for learning LangChain concepts — prompt templates, model integrations, and observability.

## Examples

### Prompt Chain — `main.py`

Builds a simple LangChain prompt chain using Claude and Langfuse for tracing.

- Feeds a bio about a person into a `PromptTemplate`
- Asks the LLM for a short summary and two interesting facts
- Traces the full chain execution to Langfuse

**Stack:** `langchain-anthropic` (claude-haiku-4-5) · `langfuse` · `python-dotenv`

---

### Groq Model Switching — `exercise-groq.py`

An exercise demonstrating how to switch between different LLM models using the LangChain-Groq integration pattern.

- Instantiates Llama 4 and Llama 3.3 models via `ChatGroq`
- Queries each model and compares responses side by side
- Uses mock objects to simulate `langchain-groq` without a live API key

**Stack:** `langchain-groq` (mocked) · Llama 4 · Llama 3.3

---

## Setup

```bash
uv sync
cp .env.example .env   # fill in your keys
```

Required environment variables:

```
ANTHROPIC_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Run

```bash
# Prompt chain example
uv run main.py

# Groq model switching exercise
uv run exercise-groq.py
```

## Branches

| Branch | Description |
|--------|-------------|
| `master` | Prompt chain + Groq exercise |
| `search-agent` | ReAct agent with Tavily web search |
