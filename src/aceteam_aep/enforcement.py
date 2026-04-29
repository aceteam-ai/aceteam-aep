"""Enforcement engine — PASS / FLAG / BLOCK decisions from safety signals.

Supports per-detector overrides with optional score thresholds::

    policy = EnforcementPolicy.from_dict({
        "default_action": "flag",
        "detectors": {
            "pii":            {"action": "block", "threshold": 0.8},
            "content_safety": {"action": "flag",  "threshold": 0.85},
            "agent_threat":   {"action": "block"},
            "cost_anomaly":   {"action": "pass",  "multiplier": 10},
        },
    })

Or from a YAML file::

    policy = EnforcementPolicy.from_yaml("aep-policy.yaml")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .safety.base import SafetyDetector, SafetySignal


@dataclass(frozen=True)
class DetectorPolicy:
    """Per-detector enforcement override."""

    action: str = "block"  # "block", "flag", "pass"
    threshold: float | None = None  # score threshold — None means always apply
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)  # detector-specific params


@dataclass(frozen=True)
class PipelinePolicy:
    """Configuration for the cascading confidence pipeline."""

    enabled: bool = False
    pass_below: float = 0.3
    block_above: float = 0.7
    confidence_threshold: float = 0.7
    layers: list[dict[str, Any]] = field(default_factory=list)
    layer_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EnforcementPolicy:
    """Configurable policy for mapping signal severities to enforcement actions.

    Two evaluation layers:
    1. Per-detector overrides (checked first) — if a signal's detector has an
       override, use that override's action (subject to threshold).
    2. Severity-based fallback — block on high, flag on medium.
    """

    block_on: frozenset[str] = frozenset({"high"})
    flag_on: frozenset[str] = frozenset({"medium"})
    allow_types: frozenset[str] = frozenset()
    default_action: str = "flag"
    overrides: dict[str, DetectorPolicy] = field(default_factory=dict)
    pipeline: PipelinePolicy = field(default_factory=PipelinePolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnforcementPolicy:
        """Build a policy from a plain dict (e.g., from JSON or inline config)."""
        overrides: dict[str, DetectorPolicy] = {}
        for name, cfg in data.get("detectors", {}).items():
            if isinstance(cfg, dict):
                overrides[name] = DetectorPolicy(
                    action=cfg.get("action", "block"),
                    threshold=cfg.get("threshold"),
                    enabled=cfg.get("enabled", True),
                    extra={
                        k: v for k, v in cfg.items() if k not in ("action", "threshold", "enabled")
                    },
                )

        pipeline_data = data.get("pipeline", {})
        pipeline = PipelinePolicy(
            enabled=pipeline_data.get("enabled", False),
            pass_below=pipeline_data.get("pass_below", 0.3),
            block_above=pipeline_data.get("block_above", 0.7),
            confidence_threshold=pipeline_data.get("confidence_threshold", 0.7),
            layers=pipeline_data.get("layers", []),
            layer_weights={
                l["name"]: l.get("weight", 1.0)
                for l in pipeline_data.get("layers", [])
                if isinstance(l, dict) and "name" in l
            },
        )

        return cls(
            block_on=frozenset(data.get("block_on", {"high"})),
            flag_on=frozenset(data.get("flag_on", {"medium"})),
            allow_types=frozenset(data.get("allow_types", set())),
            default_action=data.get("default_action", "flag"),
            overrides=overrides,
            pipeline=pipeline,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EnforcementPolicy:
        """Load policy from a YAML file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pyyaml is required for YAML policy files. "
                "Install it with: pip install aceteam-aep[yaml]"
            ) from None

        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dict, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_config(
        cls, config: EnforcementPolicy | dict[str, Any] | str | None
    ) -> EnforcementPolicy:
        """Polymorphic constructor: accepts Policy, dict, YAML path, or None."""
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        if isinstance(config, str):
            return cls.from_yaml(config)
        raise TypeError(f"Cannot create EnforcementPolicy from {type(config).__name__}")


@dataclass
class EnforcementDecision:
    """Result of evaluating safety signals against a policy."""

    action: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    reason: str = ""


# Action priority for resolving multiple signals
_ACTION_PRIORITY = {"pass": 0, "flag": 1, "block": 2}


def _resolve_signal_action(signal: SafetySignal, policy: EnforcementPolicy) -> str:
    """Determine the action for a single signal using overrides then severity fallback."""
    # Check per-detector override first
    override = policy.overrides.get(signal.signal_type) or policy.overrides.get(signal.detector)
    if override:
        if not override.enabled:
            return "pass"
        # If override has a threshold and signal has a score, check it
        if (
            override.threshold is not None
            and signal.score is not None
            and signal.score < override.threshold
        ):
            # Below threshold — downgrade to the lesser of default_action and "flag"
            return "pass" if policy.default_action == "pass" else "flag"
        return override.action

    # Severity-based fallback
    if signal.severity in policy.block_on:
        return "block"
    if signal.severity in policy.flag_on:
        return "flag"
    return "pass"


def evaluate(
    signals: Sequence[SafetySignal],
    policy: EnforcementPolicy,
) -> EnforcementDecision:
    """Evaluate safety signals against a policy and return an enforcement decision."""
    if not signals:
        return EnforcementDecision(action="pass")

    active = [s for s in signals if s.signal_type not in policy.allow_types]
    if len(active) == 0:
        return EnforcementDecision(action="pass", signals=list(signals))

    # Resolve action per signal, take highest priority
    max_action = "pass"
    reasons: list[str] = []
    for s in active:
        action = _resolve_signal_action(s, policy)
        if _ACTION_PRIORITY.get(action, 0) > _ACTION_PRIORITY.get(max_action, 0):
            max_action = action
        if action in ("block", "flag"):
            reasons.append(f"{s.signal_type}: {s.detail}")

    return EnforcementDecision(
        action=max_action,
        signals=active,
        reason="; ".join(reasons),
    )


def build_detectors_from_policy(policy: EnforcementPolicy) -> list[SafetyDetector]:
    """Create detectors with config applied from policy overrides.

    Respects enabled/disabled, threshold, and detector-specific extra params.
    """
    from .safety.agent_threat import AgentThreatDetector
    from .safety.cost_anomaly import CostAnomalyDetector

    detectors: list[SafetyDetector] = []

    # Cost anomaly
    cost_cfg = policy.overrides.get("cost_anomaly")
    if not cost_cfg or cost_cfg.enabled:
        multiplier = (cost_cfg.extra.get("multiplier", 5)) if cost_cfg else 5
        detectors.append(CostAnomalyDetector(multiplier=multiplier))

    # Agent threat
    threat_cfg = policy.overrides.get("agent_threat")
    if not threat_cfg or threat_cfg.enabled:
        detectors.append(AgentThreatDetector())

    # PII
    pii_cfg = policy.overrides.get("pii")
    if not pii_cfg or pii_cfg.enabled:
        try:
            from .safety.pii import PiiDetector

            detectors.append(PiiDetector())
        except Exception:
            pass

    # Content safety
    content_cfg = policy.overrides.get("content_safety")
    if not content_cfg or content_cfg.enabled:
        try:
            from .safety.content import ContentSafetyDetector

            threshold = (
                content_cfg.threshold if content_cfg and content_cfg.threshold is not None else 0.7
            )
            detectors.append(ContentSafetyDetector(threshold=threshold))
        except Exception:
            pass

    # FERPA (education records)
    ferpa_cfg = policy.overrides.get("ferpa")
    if ferpa_cfg and ferpa_cfg.enabled:
        from .safety.ferpa import FerpaDetector

        detectors.append(FerpaDetector())

    # Trust Engine
    te_cfg = policy.overrides.get("trust_engine")
    if te_cfg and te_cfg.enabled:
        try:
            from .safety.trust_engine import TrustEngineDetector

            te_kwargs: dict[str, Any] = {}
            if te_cfg.threshold is not None:
                te_kwargs["threshold"] = te_cfg.threshold
            dims = te_cfg.extra.get("dimensions")
            if dims:
                te_kwargs["dimensions"] = dims
            judge_url = te_cfg.extra.get("judge_service_url")
            if judge_url:
                te_kwargs["judge_service_url"] = judge_url
            model = te_cfg.extra.get("model")
            if model:
                te_kwargs["model"] = model
            base_url = te_cfg.extra.get("base_url")
            if base_url:
                te_kwargs["base_url"] = base_url
            detectors.append(TrustEngineDetector(**te_kwargs))
        except Exception:
            pass

    return detectors


def evaluate_pipeline(
    pipeline_result: Any,
    policy: EnforcementPolicy,
) -> EnforcementDecision:
    """Convert a PipelineResult into an EnforcementDecision.

    Uses the pipeline's own verdict (based on calibrated p_unsafe thresholds)
    and merges any signals raised by individual layers.
    """
    from .safety.pipeline import PipelineResult

    if not isinstance(pipeline_result, PipelineResult):
        raise TypeError(f"Expected PipelineResult, got {type(pipeline_result).__name__}")

    reasons: list[str] = []
    if pipeline_result.verdict in ("block", "flag"):
        reasons.append(
            f"pipeline: p_unsafe={pipeline_result.p_unsafe:.2f} "
            f"confidence={pipeline_result.confidence:.2f} "
            f"({pipeline_result.layers_executed} layers"
            + (f", short-circuited at {pipeline_result.short_circuited_at})" if pipeline_result.short_circuited_at else ")")
        )

    return EnforcementDecision(
        action=pipeline_result.verdict,
        signals=pipeline_result.signals,
        reason="; ".join(reasons),
    )


def build_pipeline_from_policy(
    policy: EnforcementPolicy,
    detectors: list[SafetyDetector],
) -> Any | None:
    """Build a SafetyPipeline from policy config and available detectors.

    Returns None if pipeline is not enabled in the policy.
    """
    if not policy.pipeline.enabled:
        return None

    from .safety.agent_threat import AgentThreatDetector
    from .safety.content import ContentSafetyDetector
    from .safety.custom import CustomSafetyDetector
    from .safety.pii import PiiDetector
    from .safety.pipeline import (
        ContentModelLayer,
        PawLayer,
        RegexLayer,
        SafetyPipeline,
        TrustEngineLayer,
    )
    from .safety.trust_engine import TrustEngineDetector

    detector_map: dict[str, SafetyDetector] = {d.name: d for d in detectors}
    configured_layers = {l["name"] for l in policy.pipeline.layers if isinstance(l, dict)}

    layers = []

    if "regex" in configured_layers or not configured_layers:
        regex_detectors = [
            d for d in detectors
            if isinstance(d, (PiiDetector, AgentThreatDetector))
        ]
        if regex_detectors:
            layers.append(RegexLayer(regex_detectors))

    if "paw" in configured_layers or not configured_layers:
        custom = detector_map.get("custom_safety")
        if custom and isinstance(custom, CustomSafetyDetector):
            layers.append(PawLayer(custom))

    if "content_model" in configured_layers or not configured_layers:
        content = detector_map.get("content_safety")
        if content and isinstance(content, ContentSafetyDetector):
            layers.append(ContentModelLayer(content))

    if "trust_engine" in configured_layers or not configured_layers:
        te = detector_map.get("trust_engine")
        if te and isinstance(te, TrustEngineDetector):
            layers.append(TrustEngineLayer(te))

    if not layers:
        return None

    return SafetyPipeline(
        layers=layers,
        pass_below=policy.pipeline.pass_below,
        block_above=policy.pipeline.block_above,
        confidence_threshold=policy.pipeline.confidence_threshold,
        layer_weights=policy.pipeline.layer_weights,
    )


DEFAULT_POLICY = EnforcementPolicy()


def discover_policy(explicit: str | Path | None = None) -> EnforcementPolicy:
    """Auto-discover enforcement policy using a priority chain.

    Discovery order:
    1. Explicit path argument (if provided)
    2. AEP_POLICY environment variable
    3. File discovery: aep-policy.yaml or .aep/policy.yaml walking up from CWD
    4. Built-in default policy
    """
    import os

    # 1. Explicit argument
    if explicit is not None:
        return EnforcementPolicy.from_yaml(explicit)

    # 2. Environment variable
    env_path = os.environ.get("AEP_POLICY")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return EnforcementPolicy.from_yaml(p)
        raise FileNotFoundError(f"AEP_POLICY={env_path} does not exist")

    # 3. File discovery — walk up from CWD
    home = Path.home()
    cwd = Path.cwd().resolve()
    search = cwd
    while True:
        for candidate in ("aep-policy.yaml", ".aep/policy.yaml"):
            policy_file = search / candidate
            if policy_file.is_file():
                return EnforcementPolicy.from_yaml(policy_file)
        parent = search.parent
        if parent == search or search == home:
            break
        search = parent

    # 4. Built-in default
    return DEFAULT_POLICY


__all__ = [
    "DEFAULT_POLICY",
    "DetectorPolicy",
    "EnforcementDecision",
    "EnforcementPolicy",
    "PipelinePolicy",
    "build_detectors_from_policy",
    "build_pipeline_from_policy",
    "discover_policy",
    "evaluate",
    "evaluate_pipeline",
]
