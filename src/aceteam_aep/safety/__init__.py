import logging
import os
import warnings

# Suppress HuggingFace / transformers logging noise at import time so users
# don't have to.  These run before any transformers or huggingface_hub code
# is loaded (detectors lazy-load models on first check() call).
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=".*safetensors.*")

from .agent_threat import AgentThreatDetector  # noqa: E402
from .base import DetectorRegistry, SafetyDetector, SafetySignal  # noqa: E402
from .content import ContentSafetyDetector  # noqa: E402
from .cost_anomaly import CostAnomalyDetector  # noqa: E402
from .custom import CustomPolicyStore, CustomSafetyDetector  # noqa: E402
from .pii import PiiDetector  # noqa: E402

__all__ = [
    "AgentThreatDetector",
    "ContentSafetyDetector",
    "CostAnomalyDetector",
    "CustomPolicyStore",
    "CustomSafetyDetector",
    "DetectorRegistry",
    "PiiDetector",
    "SafetyDetector",
    "SafetySignal",
]
