"""aceteam-aep: AEP-native execution layer for AI agents."""

from .agent import run_agent_loop, run_agent_loop_stream
from .budget import BudgetEnforcer, BudgetExceededError, BudgetState, ReservationToken
from .client import ChatClient
from .costs import CostNode, CostTracker
from .embeddings import CohereEmbeddings, EmbeddingClient, OllamaEmbeddings, OpenAIEmbeddings
from .envelope import Citation, ExecutionEnvelope, ExecutionError
from .envelope_builder import EnvelopeBuilder, NodeRecord
from .envelope_helpers import compute_duration, extract_primary_model, sum_cost_tree
from .factory import create_client
from .governance import (
    BudgetLimit,
    CitationClassification,
    CitationConstraints,
    GovernanceConfig,
    GovernancePolicy,
    PermissionScope,
    PromptLayer,
    SecurityLevel,
)
from .models import MODEL_REGISTRY, ModelInfo, detect_provider, get_model_info
from .pricing import DefaultPricingProvider, PricingProvider
from .prompt import wrap_context, wrap_examples, wrap_file, wrap_xml
from .protocols import HasCitations, UsageCollector
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
from .instrument import instrument, uninstrument
from .wrap import AepSession, SafetySignal, wrap

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
    # Envelope
    "Citation",
    "EnvelopeBuilder",
    "ExecutionEnvelope",
    "ExecutionError",
    "NodeRecord",
    # Envelope helpers
    "compute_duration",
    "extract_primary_model",
    "sum_cost_tree",
    # Governance
    "BudgetLimit",
    "CitationClassification",
    "CitationConstraints",
    "GovernanceConfig",
    "GovernancePolicy",
    "PermissionScope",
    "PromptLayer",
    "SecurityLevel",
    # Models
    "MODEL_REGISTRY",
    "ModelInfo",
    "detect_provider",
    "get_model_info",
    # Pricing
    "DefaultPricingProvider",
    "PricingProvider",
    # Embeddings
    "CohereEmbeddings",
    "EmbeddingClient",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
    # Prompt
    "wrap_context",
    "wrap_examples",
    "wrap_file",
    "wrap_xml",
    # Protocols
    "HasCitations",
    "UsageCollector",
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
    # Wrapper
    "wrap",
    "AepSession",
    "SafetySignal",
    # Instrumentation
    "instrument",
    "uninstrument",
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
