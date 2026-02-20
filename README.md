# aceteam-aep

AEP-native execution layer for AI agents. Direct provider SDKs and AEP protocol compliance (spans, costs, budget enforcement).

## Installation

```bash
pip install aceteam-aep
# Or with all providers:
pip install aceteam-aep[all]
```

## Quick Start

```python
from aceteam_aep import create_client, run_agent_loop, ChatMessage, tool

# Create a client
client = create_client("gpt-4o", api_key="sk-...")

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Run agent loop
result = await run_agent_loop(
    client,
    [ChatMessage(role="user", content="What is 2+2?")],
    tools=[calculator],
    system_prompt="You are a helpful assistant.",
)
```

## AEP Compliance

Every execution through `run_agent_loop` can produce AEP-compliant output:

```python
from aceteam_aep import SpanTracker, CostTracker, BudgetEnforcer

tracker = SpanTracker()
costs = CostTracker(entity="org:my-org")
budget = BudgetEnforcer(total="10.00")

result = await run_agent_loop(
    client, messages,
    span_tracker=tracker,
    cost_tracker=costs,
    budget=budget,
)

# Access AEP data
print(tracker.get_spans())      # Execution trace
print(costs.get_cost_tree())    # Hierarchical costs
print(budget.state.remaining()) # Budget remaining
```

## Streaming

```python
from aceteam_aep import run_agent_loop_stream

async for event in run_agent_loop_stream(client, messages, tools=tools):
    if event.type == "chunk":
        print(event.data["text"], end="")
    elif event.type == "tool_call_start":
        print(f"\nCalling {event.data['name']}...")
    elif event.type == "cost":
        print(f"\nCost: ${event.data['compute_cost']}")
```

## Providers

- **OpenAI** (GPT-4o, o1, o3, etc.)
- **Anthropic** (Claude Opus, Sonnet, Haiku)
- **Google** (Gemini 2.5, 3.0)
- **xAI** (Grok)
- **Ollama** (local models)
- **OpenAI-compatible** (SambaNova, TheAgentic, DeepSeek)
