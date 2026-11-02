# Safety pipeline

## Evaluation

The optional pipeline evaluates every configured layer in order. Each layer returns a `LayerResult` with `p_safe`, signals, latency, and an evaluation status. The completed scores are multiplied as a product heuristic. This product is not a calibrated probability. If a required layer is unavailable, skipped, or fails, the pipeline blocks without reporting a combined score. It does not stop early after a confident result.

| Layer | Detector coverage | Score when clean |
|-------|-------------------|------------------|
| `regex` | PII and agent threat | Configured prior, currently 0.95 |
| `paw` | Custom safety | Configured prior, currently 0.5 |
| `content_model` | Content safety | Configured prior, currently 0.9 |
| `trust_engine` | Trust Engine | Measured Trust Engine P(safe) |

Only enabled, constructed detectors can back a listed layer. Unknown, duplicate, or unavailable listed layers make policy construction fail. Registered detectors outside constructed layers, including Secret, FERPA, CostAnomaly, and third-party detectors, run once through a residual registry. Their legacy decisions are combined with the pipeline verdict by severity. Residual detector exceptions retain the registry's fail-open behavior.

A completed product at or below `1 - block_above` blocks. A product at or above `1 - pass_below` passes. Values between those thresholds flag. Detector signals can still carry their configured enforcement actions when evaluated as residuals.

## Configuration

The sample in `policies/pipeline.yaml` keeps the pipeline disabled until its section is uncommented. Its listed local layers have enabled detectors. Trust Engine is disabled in that sample. Enable its detector and configure API access before adding `trust_engine` to the layer list.

```yaml
pipeline:
  enabled: true
  pass_below: 0.3
  block_above: 0.7
  layers:
    - name: regex
    - name: paw
    - name: content_model
```

Without an enabled pipeline, the proxy evaluates the detector registry with the existing severity policy. See `src/aceteam_aep/safety/pipeline.py` for scoring and `src/aceteam_aep/enforcement.py` for policy construction and verdict combination.

## Observability

`PipelineResult` reports the combined score when complete, verdict, signals, each layer result, total latency, and an evaluation-unavailable reason when a required layer did not complete. `short_circuited_at` remains in the serialized contract and is `None` because every configured layer runs.
