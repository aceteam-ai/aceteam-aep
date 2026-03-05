"""aceteam-aep: AEP-native execution layer for AI agents."""

from .agent import run_agent_loop, run_agent_loop_stream
from .budget import BudgetEnforcer, BudgetExceededError, BudgetState, ReservationToken
from .client import ChatClient
from .costs import CostNode, CostTracker
from .embeddings import CohereEmbeddings, EmbeddingClient, OllamaEmbeddings, OpenAIEmbeddings
from .envelope import Citation, ExecutionEnvelope, ExecutionError
from .factory import create_client
from .models import MODEL_REGISTRY, ModelInfo, detect_provider, get_model_info
from .prompt import wrap_context, wrap_examples, wrap_file, wrap_xml
from .spans import Span, SpanTracker
from .stream import StreamEvent
from .structured import structured_output
from .text_splitter import split_text
from .tools import Tool, tool
from .types import (
    AgentResult,
    ChatMessage,
    ChatResponse,
    ContentBlock,
    Document,
    StreamChunk,
    ToolCallRequest,
    Usage,
)

__all__ = [
    # Agent loop
    "run_agent_loop",
    "run_agent_loop_stream",
    # Budget
    "BudgetEnforcer",
    "BudgetExceededError",
    "BudgetState",
    "ReservationToken",
    # Client
    "ChatClient",
    "create_client",
    # Costs
    "CostNode",
    "CostTracker",
    # Models
    "MODEL_REGISTRY",
    "ModelInfo",
    "detect_provider",
    "get_model_info",
    # Embeddings
    "CohereEmbeddings",
    "EmbeddingClient",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    # Envelope
    "Citation",
    "ExecutionEnvelope",
    "ExecutionError",
    # Prompt
    "wrap_context",
    "wrap_examples",
    "wrap_file",
    "wrap_xml",
    # Spans
    "Span",
    "SpanTracker",
    # Stream
    "StreamEvent",
    # Structured
    "structured_output",
    # Text splitting
    "split_text",
    # Tools
    "Tool",
    "tool",
    # Types
    "AgentResult",
    "ChatMessage",
    "ChatResponse",
    "ContentBlock",
    "Document",
    "StreamChunk",
    "ToolCallRequest",
    "Usage",
]
