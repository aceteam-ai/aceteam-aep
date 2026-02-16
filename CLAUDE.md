# CLAUDE.md - aceteam-aep

## Overview

`aceteam-aep` is the Python AEP (Agentic Execution Protocol) execution layer. It provides:

- Multi-provider LLM client abstraction (OpenAI, Anthropic, Google, xAI, Ollama)
- Agent loop with tool calling (`run_agent_loop` / `run_agent_loop_stream`)
- AEP span tracking, cost trees, budget enforcement
- ExecutionEnvelope builder
- XML prompt helpers, text splitting, embeddings

## Development Commands

```bash
uv sync --extra dev        # Install all dependencies
uv run pytest              # Run tests
uv run ruff check          # Lint
uv run ruff format         # Format
uv run pyright             # Type check
```

## Architecture

- `src/aceteam_aep/types.py` - Core types: ChatMessage, ToolCallRequest, Usage, StreamChunk
- `src/aceteam_aep/client.py` - ChatClient protocol + ChatResponse
- `src/aceteam_aep/factory.py` - `create_client()` provider detection from model name
- `src/aceteam_aep/agent.py` - `run_agent_loop()` / `run_agent_loop_stream()`
- `src/aceteam_aep/spans.py` - SpanTracker
- `src/aceteam_aep/costs.py` - CostTracker with hierarchical cost trees
- `src/aceteam_aep/budget.py` - BudgetEnforcer with pessimistic reservation
- `src/aceteam_aep/envelope.py` - ExecutionEnvelope builder
- `src/aceteam_aep/tools.py` - Tool protocol + @tool decorator
- `src/aceteam_aep/prompt.py` - XML wrapping helpers
- `src/aceteam_aep/providers/` - Provider implementations

## Conventions

- Use `uv` for all dependency management
- All public functions have type hints
- Tests use pytest-asyncio with `asyncio_mode = "auto"`
- Decimal arithmetic for all cost/budget calculations
