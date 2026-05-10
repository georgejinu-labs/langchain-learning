# LangChain Search Agent

A ReAct agent built with LangGraph that uses Claude as the LLM and Tavily for web search, with full observability via Langfuse.

## What it does

The agent answers questions by searching the web in real time. Given a query like *"What is the weather in NYC?"*, it:

1. Decides to call the Tavily search tool
2. Runs the search and reads the results
3. Returns a natural language answer

## Stack

| Component | Library |
|-----------|---------|
| LLM | `claude-opus-4-5` via `langchain-anthropic` |
| Agent loop | `langgraph` — `create_react_agent` |
| Search tool | `langchain-tavily` — `TavilySearch` |
| Tracing | `langfuse` — `CallbackHandler` |

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your keys
cp .env.example .env
```

Required environment variables in `.env`:

```
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Run

```bash
uv run main.py
```

Traces appear in your [Langfuse dashboard](https://cloud.langfuse.com) within seconds of each run.
