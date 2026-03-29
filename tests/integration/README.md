# Integration Tests

End-to-end tests that verify AEP works with real agent frameworks.
These tests make actual LLM API calls through the AEP proxy.

## Requirements

- `OPENAI_API_KEY` environment variable set
- Each framework installed (DSPy, CrewAI, LangChain)

## Running

```bash
# All integration tests
uv run pytest tests/integration/ -v

# Specific framework
uv run pytest tests/integration/test_dspy.py -v
uv run pytest tests/integration/test_crewai.py -v
uv run pytest tests/integration/test_langchain.py -v

# Skip if no API key
uv run pytest tests/integration/ -v -k "not integration"
```

## What Each Test Verifies

1. Calls route through the AEP proxy (dashboard state shows them)
2. Cost is tracked (non-zero after call)
3. Safety detectors fire on PII input (BLOCK or signal raised)
4. Normal calls pass through with receipts
