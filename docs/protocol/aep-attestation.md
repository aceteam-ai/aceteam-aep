# AEP-Attestation — Signed Verdicts and Authenticated Audit Chains

**Status:** Draft
**Date:** 2026-03-29 (format_version 2 added 2026-09-12, aceteam-ai/aceteam#9657)
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

L1 has two statement formats:

- **format_version 1** (original, 2026-03-29): signs a narrow 5-field verdict (`call_id`, `action`, `signals`, `timestamp`, `confidence`). Chain position (`chain_height`/`chain_hash`/`prev_chain_hash`) and the signer's session identity live **outside** the signed bytes.
- **format_version 2** (current, 2026-09-12): signs a complete statement that additionally binds schema/domain, signer identity, session/execution identity, sequence position, and the previous *authenticated* link. This is the format `AttestationEngine.sign_verdict` produces and `verify_chain`/`verify_chain_report` check today.

format_version 1 is documented in §3–§4 and §6.1–§6.3 below for reference and because every format_version 2 entry still carries the equivalent format_version 1 fields unchanged (`verdict_hash`/`signature`/`chain_hash`) for callers that only need the narrow single-verdict guarantee. **A format_version 1 chain does not satisfy the strict full-chain guarantee in §8.2** — see §10 for why, and §5 / §6.5–§6.8 for the format_version 2 statement that replaces it.

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

## 3. Verdict Hash (format_version 1, narrow — `verdict_hash()` / `verify_verdict()`)

The verdict hash is a deterministic digest of the verdict data. It is the value that gets signed.

**Narrow, documented meaning:** a valid signature over this hash proves only "the holder of the signing key produced this exact 5-field verdict." It says nothing about the verdict's position in a sequence, which execution/session produced it, or whether entries around it were omitted or reordered. Use §5–§6.8 (format_version 2) for the full authenticated-statement guarantee. `verify_verdict()` retains this narrow meaning permanently — it is a useful primitive, not a chain guarantee, and other code must not treat it as one.

### 3.1 Canonicalization Contract

Both format_version 1 (`verdict_hash`) and format_version 2 (`statement_digest`) use the same canonicalization contract (`canonical_json`), so a drift in one cannot silently diverge from the other:

1. UTF-8 output; `ensure_ascii=true` (non-ASCII escaped as `\uXXXX`).
2. Object keys sorted by Unicode code point.
3. Compact separators: `,` and `:`, no whitespace, no trailing commas.
4. Object members whose value is `null` are omitted entirely. `null` inside a list is retained (list position is meaningful).
5. Numbers as decimal, no scientific notation. Floats **must be finite** — `NaN`, `Infinity`, and `-Infinity` are rejected at both sign time and verify time rather than serialized as non-standard JSON tokens different implementations would canonicalize differently.
6. Booleans and integers serialize as their normal JSON types (a bare `sequence`/`chain_height` integer is never accepted where a boolean was supplied, even though `bool` is a subtype of `int` in some languages).

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

### 4.4 Execution/Session Identity (format_version 2)

Each `AttestationEngine` instance additionally has an `execution_id` — a random identifier generated once per engine instance (by default, once per proxy process/session). It is bound into every statement that engine signs. A new engine instance always gets a distinct `execution_id`; two independently created engines using the *same signing key* still produce chains that cannot be spliced into each other (§6.7, §8.2).

## 5. Response Headers

Signed verdicts are transmitted as HTTP response headers. Both format_version 1 (legacy, narrow) and format_version 2 (authenticated statement) headers are included in every signed response, so an HTTP-only consumer archiving response headers has everything needed to reconstruct and verify the complete statement — not only a caller with in-process access to `AttestationEngine.chain`:

```http
X-AEP-Enforcement: pass
X-AEP-Call-Id: a1b2c3d4
X-AEP-Verdict-Hash: sha256:7f8a9b...
X-AEP-Signature: ed25519:1a2b3c...
X-AEP-Signer-Id: proxy:prod-01
X-AEP-Chain-Height: 47
X-AEP-Chain-Hash: sha256:8e9f0a...

X-AEP-Format-Version: 2
X-AEP-Execution-Id: 9f86d0...
X-AEP-Sequence: 47
X-AEP-Timestamp: 2026-09-12T00:00:00+00:00
X-AEP-Signals: [{"signal_type":"pii","severity":"low"}]
X-AEP-Confidence: 0.73
X-AEP-Prev-Link: sha256:...
X-AEP-Statement-Digest: sha256:...
X-AEP-Statement-Signature: ed25519:...
```

`X-AEP-Confidence` is present only when the verdict has a confidence score (its absence, like a `null` field in the canonical JSON, means no confidence was supplied — not zero confidence). When signing is not enabled, all attestation headers are omitted.

## 6. Audit Chain

### 6.1 format_version 1 Chain Structure (legacy, unsigned links)

Verdicts are chained sequentially. Each verdict's chain hash includes the previous chain hash:

```
Chain hash for verdict N:
  chain_hash_N = SHA-256(chain_hash_{N-1} || verdict_hash_N)

Genesis:
  chain_hash_0 = SHA-256("aep-genesis" || verdict_hash_0)
```

Where `||` is byte concatenation. **This computation uses only publicly known values (the previous chain hash and the current verdict hash), so anyone can recompute a valid-looking `chain_hash`/`prev_chain_hash` for a chain they have edited, without the signing key.** `chain_height`, `chain_hash`, and `prev_chain_hash` are retained on every entry for backward compatibility and cosmetic display (e.g. the dashboard's "latest hash"), but as of format_version 2 they are **not** used to make any authentication decision. See §10.

### 6.2 format_version 1 Properties (as originally believed — since narrowed)

-~~Tamper evidence: changing any verdict breaks all subsequent chain hashes~~ — false if the presenter is allowed to recompute the unsigned links after editing (§10).
- ~~Ordering proof~~, ~~Completeness~~ — same caveat.

### 6.3 format_version 1 Chain Metadata

| Field | Type | Description |
|-------|------|-------------|
| `chain_height` | integer | Monotonically increasing, starts at 0 (unauthenticated) |
| `chain_hash` | string | `sha256:` prefixed hex digest (unauthenticated) |
| `prev_chain_hash` | string | Previous chain hash, empty string for genesis (unauthenticated) |

### 6.4 Session Scope

A chain is scoped to a proxy session (process lifetime; format_version 2 additionally scopes it to `execution_id`). Chain height resets when the proxy restarts, and a new `execution_id` is generated. For persistent chains across restarts, the proxy should store and load the latest chain state, or treat each process lifetime as its own verifiable segment.

### 6.5 format_version 2: The Signed Statement

A format_version 2 statement binds every field that a presenter could otherwise mutate on a *displayed* verdict without the signing key:

```
Statement = {
    schema:             "aceteam.aep.verdict-statement"
    format_version:     2
    signer_id:          string           // §4.3
    execution_id:       string           // §4.4 — this session's identity
    call_id:            string
    action:             "pass" | "flag" | "block"
    signals:            Signal[]
    confidence:         float | null
    timestamp:          ISO 8601
    sequence:           integer          // this entry's position, 0-based
    prev_link:          string           // sha256: digest of the PREVIOUS
                                          // entry's statement (authenticated,
                                          // not the unsigned §6.1 chain_hash)
    input_commitment:   string | null    // optional caller-supplied reference
    output_commitment:  string | null    // (e.g. a content hash), bound when
    policy_commitment:  string | null    // supplied — see §6.6
}

statement_digest    = SHA-256(canonical_json(Statement))   // §3.1 contract
statement_signature = Ed25519_Sign(private_key, statement_digest_bytes)
```

Genesis (`sequence == 0`) uses a `prev_link` bound to the session, not a fixed constant:

```
prev_link_0 = SHA-256("aep-genesis-v2:" || execution_id)
```

Binding genesis to `execution_id` is what makes position-0 entries from two different sessions distinguishable even when signed by the same key (§6.7).

### 6.6 Content and Policy Commitments

`input_commitment`, `output_commitment`, and `policy_commitment` are optional, caller-supplied references (typically content hashes) bound into the statement when supplied. **Their absence means "unbound," never "content/policy was checked."** A verifier can report whether a chain's commitments match an expected value, but that comparison only happened for entries where a commitment was actually supplied and actually compared — this specification does not yet define which commitments are mandatory for a given delegation profile (tracked separately in aceteam-ai/aceteam#9652 S4).

### 6.7 Session/Execution Binding

Every entry in one presented chain must share exactly one `execution_id`, and `prev_link` must chain to the *actual, previously verified* statement digest — not a value the presenter can compute independently. Consequences:

- A valid entry from session A can never be accepted as part of session B's chain, even if both were signed by the identical private key.
- A verifier may additionally pin an expected `execution_id` up front (`verify_chain(..., expected_execution_id=...)` / CLI `--execution-id`) to reject an entire chain that belongs to the wrong session outright.

### 6.8 What a Valid Chain Does and Does Not Prove

A `"valid"` result from §8.2 proves the presented entries are an unbroken, authenticated sequence: every statement's digest and signature check out, sequence positions are contiguous from 0, and each entry's `prev_link` matches its predecessor's actual signed digest.

**It does not prove that sequence is the complete history.** A presenter can still legitimately withhold everything after some point, or (for a chain persisted externally, e.g. as a JSONL export) present a truncated prefix. Detecting truncation or a forked/conflicting history requires a previously trusted checkpoint external to the chain itself — an expected final `sequence`/`statement_digest`/`execution_id` obtained out of band — not something this specification adds automatically. Certificate Transparency's signed checkpoints and consistency proofs are a useful precedent for this problem ([RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html)); it likewise does not prove unobserved events never occurred. This slice does not add a public transparency service — only the authenticated-statement primitive a checkpoint scheme would sit on top of.

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

### 8.1 Single Verdict (narrow primitive, either format)

```
1. Fetch public key for X-AEP-Signer-Id from /.well-known/aep-keys.json
2. Decode X-AEP-Verdict-Hash (remove sha256: prefix, hex decode)
3. Decode X-AEP-Signature (remove ed25519: prefix, hex decode)
4. Ed25519_Verify(public_key, verdict_hash_bytes, signature_bytes)
5. If valid: this exact 5-field verdict was signed by this key (§3 — narrow meaning only)
```

Never raises: malformed hex, wrong-length signatures, or other crypto-library errors are treated as "does not verify," not as an exception.

### 8.2 Chain Verification (strict, format_version 2)

```
1. Obtain all entries E_0 .. E_N (from archived response headers or the
   engine's exported chain).
2. If ANY entry lacks a complete format_version 2 statement (schema,
   format_version, signer_id, execution_id, call_id, action, signals,
   timestamp, sequence, prev_link, statement_digest, statement_signature):
     -> status = "legacy_unsupported". Not evidence of tampering — the
        older format's bytes cannot honestly be reinterpreted under this
        guarantee. Do not silently upgrade it.
3. All entries must share exactly one execution_id (§6.7); if an expected
   execution_id was supplied, it must match.
4. For each E_i (i = 0..N):
   a. sequence must equal i exactly (an int, not a bool).
   b. prev_link must equal:
        - genesis_link(execution_id)          if i == 0   (§6.5)
        - statement_digest recomputed for E_{i-1}   otherwise
      (never a value read directly off E_i's own presented fields).
   c. Recompute statement_digest from E_i's presented fields (§3.1, §6.5)
      — never trust a presented "statement_digest" field for this decision.
   d. Ed25519_Verify(public_key, recomputed_digest, E_i.statement_signature)
      must succeed.
   Any failure -> status = "invalid", with the first failing index and a
   reason (distinguishing a sequence/link problem, which is caught before
   any signature check, from a digest/signature mismatch).
5. If every entry passes: status = "valid" — see §6.8 for what this does
   and does not prove.
```

`verify_chain()` returns a plain boolean for convenience; `verify_chain_report()` returns the full `ChainVerificationResult` (`valid`, `status`, `reason`, `failed_index`) so callers and the CLI can distinguish "unsupported" from "tampered."

### 8.3 CLI Verification

```bash
aceteam-aep verify --pub-key ./aep.pub --chain audit-log.jsonl [--execution-id <id>]
```

Exit codes: `0` for `"valid"` (also for an empty chain), non-zero for `"legacy_unsupported"` or `"invalid"`, each printed with a distinct message and, where applicable, the failing entry index and reason.

## 9. Conformance

An AEP-Attestation L1 conformant proxy MUST:

1. Sign every verdict with Ed25519 when `--sign-key` is provided, producing both the format_version 1 narrow fields (backward compatibility) and the format_version 2 authenticated statement.
2. Include the format_version 1 and format_version 2 attestation headers (§5) in every signed response, including `signals` and `timestamp` — omitting fields needed to recompute the statement digest from the transport is a conformance failure (this is the exact defect fixed by aceteam-ai/aceteam#9657).
3. Generate a distinct `execution_id` per engine instance and never reuse one across sessions.
4. Maintain a monotonically increasing, contiguous `sequence` per session.
5. Serve public keys at `/.well-known/aep-keys.json`.
6. Use the canonicalization contract in §3.1 for both `verdict_hash` and `statement_digest`, rejecting non-finite floats and malformed field types rather than silently reinterpreting them.
7. Verifiers MUST report format_version 1 (or otherwise incomplete) chains as `"legacy_unsupported"` rather than upgrading them to the format_version 2 guarantee, and MUST NOT make an authentication decision based on `chain_hash`/`prev_chain_hash`/`chain_height` alone.

## 10. Security Considerations

- **The original chain-position gap (fixed in format_version 2, aceteam-ai/aceteam#9657):** in format_version 1, `chain_height`/`chain_hash`/`prev_chain_hash` are computed from publicly known values, and the exported entry did not carry `signals`/`confidence` at all. A presenter could therefore (a) change a displayed `action` (or any other field the narrow `verdict_hash` didn't happen to still match) and the chain would still "verify," and (b) drop an interior entry and recompute the remaining unsigned links without ever touching the signing key. format_version 2 closes both: every displayed field is inside the signed statement, and the chain link (`prev_link`) is the *previous authenticated digest*, not a recomputable hash.
- **Legacy chains are not silently upgraded.** A format_version 1 chain is reported `"legacy_unsupported"`, distinct from `"invalid"` — it is not evidence of tampering, only insufficient authenticated material for the stronger guarantee.
- **A valid chain is not proof of a complete history.** See §6.8. Detecting truncation needs an externally trusted checkpoint.
- **Cross-session splicing.** Two engines using the same signing key still cannot have their chains spliced together — each session's `execution_id` and genesis `prev_link` differ, and mixing entries from two sessions is detected before any digest/signature check runs.
- **Commitments are opt-in.** `input_commitment`/`output_commitment`/`policy_commitment` are unbound unless supplied by the caller; their absence is never proof that content or policy was checked (§6.6).
- **Key compromise:** if the private key is stolen, an attacker can forge verdicts (of either format). Mitigate with key rotation and short validity periods.
- **Replay attacks:** contiguous `sequence` (format_version 2) or monotonic `chain_height` (format_version 1) prevents replaying old verdicts undetected within one session.
- **Clock skew:** timestamps are informational and bound into the format_version 2 statement (so tampering with a timestamp is detectable), but chain ordering is by `sequence`, not by time.
- **Proxy compromise:** L1 trusts the proxy. If the proxy is compromised, it can sign false verdicts. L2 (detector attestation) mitigates this by requiring each detector to sign independently.
- **Signatures are not correctness claims.** A verified statement proves what was signed and by whom, and (for format_version 2) where it sits in an authenticated sequence. It does not prove the underlying safety evaluation was correct, complete, or that every real action was logged.
