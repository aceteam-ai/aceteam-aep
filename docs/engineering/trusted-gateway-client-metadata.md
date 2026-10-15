# Trusted gateway call metadata

Gateway correlation is opt-in. Configure `trusted_gateway_url` when creating an
OpenAI-compatible or Anthropic client. A regular `base_url`, including one that
points to a gateway, does not enable AEP header handling.

```python
from aceteam_aep import AepRequestContext, create_client, current_tool_origin, run_agent_loop

client = create_client(
    "gpt-4o",
    api_key="gateway-key",
    trusted_gateway_url="https://gateway.example/v1",
)
context = AepRequestContext(trace_id="trace-123", entity="org:example")
response = await client.chat_with_context(messages, request_context=context)
call_id = response.response_metadata.call_id if response.response_metadata else None

# The agent loop accepts the same context and scopes each tool's origin.
result = await run_agent_loop(client, messages, tools=tools, request_context=context)
for metadata in result.response_metadata:
    print(metadata.call_id)

# Inside a tool callback, current_tool_origin() refers to the response that
# proposed that tool call. It is None outside the callback.
```

The optional `AepRequestContext` sends `X-AEP-Trace-ID` and `X-AEP-Entity` only
through this configured transport. Invalid values are rejected before sending.
The common `ChatClient` protocol has no new required methods or arguments;
third-party clients continue to work. Passing `request_context` to an agent
loop requires a client with the optional `ContextChatClient` capability.

`AepResponseMetadata` is immutable and contains only validated, printable,
bounded `X-AEP-Call-ID`, `X-AEP-Trace-ID`, and `X-AEP-Entity` header values.
The call ID is opaque and copied exactly from the trusted gateway response.
Missing or invalid values remain `None`; provider response IDs, tool IDs, and
local span IDs are never used as substitutes. Direct provider clients ignore
similarly named response headers and never send AEP context headers.

For streaming calls, the first `StreamChunk` contains the header metadata before
SSE parsing. Its `response_complete` is false. A final metadata chunk with
`response_complete=True` appears only after the stream completes successfully.
Agent streams emit matching `response_metadata` events with a `complete` flag,
and tool events carry the proposal's metadata. If parsing or SSE iteration
fails after headers arrive, `GatewayResponseError.response_metadata` retains
the captured values and `response_complete` is false. A cancelled consumer
can retain the first metadata chunk or event it received. Completion reflects
the body lifecycle, not a gateway verdict.

Completed `ToolCallRequest` proposals also carry the immutable metadata in
their optional `origin` field. A tool callback can read the same origin through
`current_tool_origin()`; it resets after the callback, including on error or
cancellation.

Correlation is for tracing calls across systems. It is neither a safety
decision nor a cryptographic attestation.
