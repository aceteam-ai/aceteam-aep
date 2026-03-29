# AEP-Attestation — Signed Verdicts and Merkle Audit Chains

**Status:** Draft
**Date:** 2026-03-29
**Authors:** AceTeam Engineering
**License:** Apache 2.0

---

## 1. Overview

AEP-Attestation defines how safety verdicts are cryptographically signed and chained into tamper-evident audit trails. It answers the enterprise question: "How do we know the safety checks actually ran?"

Three levels of attestation, each building on the previous:

| Level | What | Trust Boundary |
|-------|------|---------------|
| **L1: Proxy Signing** | Proxy signs aggregate verdict | Trust the proxy operator |
| **L2: Detector Attestation** | Each detector independently signs | Trust individual detectors |
| **L3: Third-Party Verification** | External auditor certifies the chain | Trust the auditor |

This specification covers L1. L2 and L3 are follow-ups.

## 2. Verdict Structure

A verdict is the output of evaluating safety signals against an enforcement policy.

```
Verdict = {
    call_id:    string        // unique per LLM call
    action:     "pass" | "flag" | "block"
    signals:    Signal[]      // detector outputs that informed the decision
    timestamp:  ISO 8601      // when the verdict was issued
    confidence: float | null  // calibrated confidence score (0.0 - 1.0), if available
}
```

Each signal:

```
Signal = {
    signal_type:  string      // "pii", "toxicity", "agent_threat", "cost_anomaly", ...
    severity:     "high" | "medium" | "low"
    detail:       string      // human-readable description
    score:        float | null // detector confidence, if available
    detector:     string      // detector name/version
}
```

## 3. Verdict Hash

The verdict hash is a deterministic digest of the verdict data. It is the value that gets signed.

### 3.1 Canonical Form

To ensure deterministic hashing regardless of implementation language:

1. Construct a JSON object with fields in alphabetical order
2. Use minimal encoding (no whitespace, no trailing commas)
3. Numbers as decimal (no scientific notation)
4. Null fields omitted

```json
{"action":"pass","call_id":"a1b2c3","confidence":0.73,"signals":[{"detail":"clean","score":0.92,"severity":"low","signal_type":"pii"}],"timestamp":"2026-03-29T10:00:00Z"}
```

### 3.2 Hash Algorithm

```
verdict_hash = SHA-256(canonical_json_bytes)
```

Encoded as lowercase hex with `sha256:` prefix:

```
sha256:7f8a9b3c2d1e0f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f
```

## 4. Signing

### 4.1 Algorithm

Ed25519 (RFC 8032). Chosen for:
- Fast signing (~60μs) — negligible overhead on the hot path
- Small signatures (64 bytes)
- Small keys (32 bytes public, 64 bytes private)
- No configuration parameters (no curve choices, no hash choices)
- Widely implemented (Go, Python, Rust, Node.js standard libraries)

### 4.2 Signature

```
signature = Ed25519_Sign(private_key, verdict_hash_bytes)
```

Encoded as lowercase hex with `ed25519:` prefix:

```
ed25519:1a2b3c4d5e6f...
```

### 4.3 Signer Identity

Each signer has a unique ID:

```
signer_id = "proxy:<deployment-name>"
```

Examples: `proxy:prod-01`, `proxy:staging`, `proxy:safeclaw-demo`

## 5. Response Headers

Signed verdicts are transmitted as HTTP response headers:

```http
X-AEP-Enforcement: pass
X-AEP-Confidence: 0.73
X-AEP-Call-Id: a1b2c3d4
X-AEP-Verdict-Hash: sha256:7f8a9b...
X-AEP-Signature: ed25519:1a2b3c...
X-AEP-Signer-Id: proxy:prod-01
X-AEP-Chain-Height: 47
X-AEP-Chain-Hash: sha256:8e9f0a...
```

When signing is not enabled, `X-AEP-Verdict-Hash`, `X-AEP-Signature`, `X-AEP-Signer-Id`, `X-AEP-Chain-Height`, and `X-AEP-Chain-Hash` are omitted.

## 6. Merkle Audit Chain

### 6.1 Chain Structure

Verdicts are chained sequentially. Each verdict's chain hash includes the previous chain hash, creating a tamper-evident sequence.

```
Chain hash for verdict N:
  chain_hash_N = SHA-256(chain_hash_{N-1} || verdict_hash_N)

Genesis:
  chain_hash_0 = SHA-256("aep-genesis" || verdict_hash_0)
```

Where `||` is byte concatenation.

### 6.2 Properties

- **Tamper evidence:** changing any verdict breaks all subsequent chain hashes
- **Ordering proof:** verdicts are sequenced by chain height
- **Completeness:** gaps in the chain are detectable (missing height breaks the chain)
- **Efficient verification:** verify from any point by recomputing forward

### 6.3 Chain Metadata

| Field | Type | Description |
|-------|------|-------------|
| `chain_height` | integer | Monotonically increasing, starts at 0 |
| `chain_hash` | string | `sha256:` prefixed hex digest |
| `prev_chain_hash` | string | Previous chain hash (empty string for genesis) |

### 6.4 Session Scope

A chain is scoped to a proxy session (process lifetime). Chain height resets when the proxy restarts. For persistent chains across restarts, the proxy should store and load the latest chain state.

## 7. Key Management

### 7.1 Key Generation

```bash
aceteam-aep keygen [--output ./aep-keys/]
```

Generates:
- `aep.key` — Ed25519 private key (PEM or raw 64 bytes)
- `aep.pub` — Ed25519 public key (PEM or raw 32 bytes)

### 7.2 Key Distribution

Public keys are discoverable at a well-known URL on the proxy:

```
GET /.well-known/aep-keys.json
```

Response:

```json
{
  "keys": [
    {
      "signer_id": "proxy:prod-01",
      "algorithm": "ed25519",
      "public_key": "base64:...",
      "valid_from": "2026-03-01T00:00:00Z",
      "valid_until": "2027-03-01T00:00:00Z"
    }
  ]
}
```

Multiple keys support rotation. Verifiers match `X-AEP-Signer-Id` to the key list.

### 7.3 Key Rotation

- New key added to `aep-keys.json` with future `valid_from`
- Proxy switches to new key at `valid_from`
- Old key remains in `aep-keys.json` until `valid_until` passes
- Verifiers accept signatures from any key valid at the verdict's timestamp

## 8. Verification

### 8.1 Single Verdict

```
1. Fetch public key for X-AEP-Signer-Id from /.well-known/aep-keys.json
2. Decode X-AEP-Verdict-Hash (remove sha256: prefix, hex decode)
3. Decode X-AEP-Signature (remove ed25519: prefix, hex decode)
4. Ed25519_Verify(public_key, verdict_hash_bytes, signature_bytes)
5. If valid: verdict is authentic
```

### 8.2 Chain Verification

```
1. Obtain all verdicts V_0 through V_N (from audit log or API)
2. For each V_i:
   a. Verify signature (per 8.1)
   b. Recompute chain_hash_i = SHA-256(chain_hash_{i-1} || verdict_hash_i)
   c. Compare to stored chain_hash_i
3. If all match: chain is intact, no tampering
```

### 8.3 CLI Verification

```bash
aceteam-aep verify --pub-key ./aep.pub --chain audit-log.jsonl
```

## 9. Conformance

An AEP-Attestation L1 conformant proxy MUST:

1. Sign every verdict with Ed25519 when `--sign-key` is provided
2. Include all six attestation headers in every response
3. Maintain a monotonically increasing chain height per session
4. Serve public keys at `/.well-known/aep-keys.json`
5. Use the canonical JSON form for verdict hashing (Section 3.1)
6. Use SHA-256 for both verdict hash and chain hash

## 10. Security Considerations

- **Key compromise:** if the private key is stolen, an attacker can forge verdicts. Mitigate with key rotation and short validity periods.
- **Replay attacks:** chain height prevents replaying old verdicts (height must be monotonic).
- **Clock skew:** timestamps are informational, not used in hash computation. Chain ordering is by height, not time.
- **Proxy compromise:** L1 trusts the proxy. If the proxy is compromised, it can sign false verdicts. L2 (detector attestation) mitigates this by requiring each detector to sign independently.
