"""AEP Attestation — Ed25519 signed verdicts and Merkle audit chains.

Implements AEP-Attestation Level 1 (proxy signing):

1. Deterministic verdict hashing (canonical JSON → SHA-256)
2. Ed25519 signing of verdict hashes
3. Merkle audit chain (each verdict chains to previous)
4. Key generation and verification

Usage::

    from aceteam_aep.attestation import AttestationEngine, generate_keypair

    # Generate keys
    private_key, public_key = generate_keypair()
    private_key.save("aep.key")
    public_key.save("aep.pub")

    # Sign verdicts
    engine = AttestationEngine(private_key, signer_id="proxy:prod-01")
    headers = engine.sign_verdict(
        call_id="abc123",
        action="pass",
        signals=[],
        confidence=0.73,
    )
    # headers: {X-AEP-Verdict-Hash, X-AEP-Signature, X-AEP-Signer-Id,
    #           X-AEP-Chain-Height, X-AEP-Chain-Hash}

    # Verify
    from aceteam_aep.attestation import verify_verdict
    valid = verify_verdict(verdict_hash, signature, public_key)

See docs/protocol/aep-attestation.md for the full specification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Ed25519 via cryptography library (transitive dep from httpx/anthropic)
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(
            "cryptography library required for attestation. Install with: pip install cryptography"
        )


# ---------------------------------------------------------------------------
# Canonical JSON hashing
# ---------------------------------------------------------------------------


def canonical_json(data: dict[str, Any]) -> bytes:
    """Produce deterministic canonical JSON bytes.

    Canonicalization contract (see docs/protocol/aep-attestation.md §3.1):

    - UTF-8 output, ``ensure_ascii=True`` (non-ASCII escaped as ``\\uXXXX``).
    - Object keys sorted by Unicode code point.
    - Compact separators: ``,`` and ``:``, no whitespace.
    - Object members whose value is ``None`` are omitted entirely (``null``
      inside a list is retained, since list position is meaningful).
    - Floats must be finite; ``NaN``/``Infinity``/``-Infinity`` are rejected
      rather than silently serialized as non-standard JSON tokens that
      different implementations would canonicalize differently.
    - Booleans and ints are serialized as their normal JSON types.

    Raises ``ValueError`` for non-finite floats, ``TypeError`` for values
    that are not JSON-serializable under this contract.
    """

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in sorted(obj.items()) if v is not None}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError(f"non-finite float in canonical payload: {obj!r}")
        return obj

    cleaned = _clean(data)
    return json.dumps(cleaned, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def verdict_hash(
    *,
    call_id: str,
    action: str,
    signals: list[dict[str, Any]],
    timestamp: str,
    confidence: float | None = None,
) -> str:
    """Compute deterministic SHA-256 hash of a verdict.

    Returns ``sha256:<hex>`` prefixed string.

    Narrow, documented meaning: this is a digest over five verdict fields
    only. A signature over this hash (see :func:`verify_verdict`) proves
    only "the holder of the signing key produced this exact 5-field
    verdict"; it says nothing about the verdict's position in a sequence,
    which execution/session produced it, or whether any entries around it
    were omitted. Use :func:`AttestationEngine.sign_verdict` /
    :func:`verify_chain` for the full authenticated-statement guarantee.
    """
    data = {
        "call_id": call_id,
        "action": action,
        "signals": signals,
        "timestamp": timestamp,
        "confidence": confidence,
    }
    digest = hashlib.sha256(canonical_json(data)).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Versioned signed statement (format_version 2)
# ---------------------------------------------------------------------------
#
# format_version 1 (the original ``verdict_hash``/merkle-chain design above)
# signs only {call_id, action, signals, timestamp, confidence}. Chain
# position (chain_height/chain_hash/prev_chain_hash) and the signer's
# session identity live outside the signed bytes, so a presenter can swap
# an entry's displayed fields, or drop an entry and recompute the unsigned
# chain links, without invalidating any individual signature. See
# aceteam-ai/aceteam#9657 / #9652 Appendix A for the reproduction.
#
# format_version 2 fixes this by folding everything the audit presentation
# claims into the bytes that get signed: schema/domain, signer identity,
# session/execution identity, call id, action, signals, confidence, an
# explicit sequence position, and the *previous authenticated statement's
# digest* (not an unsigned recomputable hash). verify_chain_report()
# recomputes every field from the presented entry and never trusts a
# presented digest/signature/chain-position field without doing so.

STATEMENT_SCHEMA = "aceteam.aep.verdict-statement"
STATEMENT_VERSION = 2

_GENESIS_PREFIX = b"aep-genesis-v2:"

# The exact set of fields bound into a format_version 2 statement, in the
# order build_statement() assembles them (canonical_json re-sorts keys
# regardless, but this is the allow-list used to *extract* a statement back
# out of a presented chain entry for verification).
_STATEMENT_FIELDS = (
    "schema",
    "format_version",
    "signer_id",
    "execution_id",
    "call_id",
    "action",
    "signals",
    "confidence",
    "timestamp",
    "sequence",
    "prev_link",
    "input_commitment",
    "output_commitment",
    "policy_commitment",
)


def _genesis_link(execution_id: str) -> str:
    """The authenticated prev_link value for position 0 of a session's chain.

    Binding genesis to execution_id means the first entry of one session
    cannot be presented as a valid first entry of another session, even
    when both were signed by the same key.
    """
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("execution_id must be a non-empty string")
    digest = hashlib.sha256(_GENESIS_PREFIX + execution_id.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_statement(
    *,
    signer_id: str,
    execution_id: str,
    call_id: str,
    action: str,
    signals: list[dict[str, Any]],
    timestamp: str,
    sequence: int,
    prev_link: str,
    confidence: float | None = None,
    input_commitment: str | None = None,
    output_commitment: str | None = None,
    policy_commitment: str | None = None,
) -> dict[str, Any]:
    """Assemble the fields bound into a format_version 2 signed statement.

    Every field an attacker could change on a *displayed* verdict without
    the signing key — the action, the call id, the signals, the confidence,
    the timestamp, which signer/session produced it, where it sits in the
    sequence, and what it chains from — is bound here. Raises ``ValueError``
    for malformed input rather than silently producing bytes that would
    canonicalize ambiguously.

    Pure and deterministic: same inputs always produce the same dict (key
    order does not matter — :func:`canonical_json` re-sorts).
    """
    if not isinstance(signer_id, str) or not signer_id:
        raise ValueError("signer_id must be a non-empty string")
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("execution_id must be a non-empty string")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("call_id must be a non-empty string")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string")
    if not isinstance(signals, list) or not all(isinstance(s, dict) for s in signals):
        raise ValueError("signals must be a list of dicts")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("timestamp must be a non-empty string")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative int")
    if not isinstance(prev_link, str) or not prev_link:
        raise ValueError("prev_link must be a non-empty string")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric or null")
        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise ValueError(f"confidence must be finite, got {confidence!r}")
    for name, value in (
        ("input_commitment", input_commitment),
        ("output_commitment", output_commitment),
        ("policy_commitment", policy_commitment),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")

    return {
        "schema": STATEMENT_SCHEMA,
        "format_version": STATEMENT_VERSION,
        "signer_id": signer_id,
        "execution_id": execution_id,
        "call_id": call_id,
        "action": action,
        "signals": signals,
        "confidence": confidence,
        "timestamp": timestamp,
        "sequence": sequence,
        "prev_link": prev_link,
        "input_commitment": input_commitment,
        "output_commitment": output_commitment,
        "policy_commitment": policy_commitment,
    }


def statement_digest(statement: dict[str, Any]) -> str:
    """SHA-256 digest of a canonicalized format_version 2 statement."""
    digest = hashlib.sha256(canonical_json(statement)).hexdigest()
    return f"sha256:{digest}"


def _is_v2_entry(entry: Any) -> bool:
    """Whether a presented chain entry carries a complete v2 statement.

    Entries lacking these fields (format_version 1, or hand-rolled/garbage
    data) are "legacy" — they may still satisfy the narrow
    ``verify_verdict`` primitive, but they do not carry enough authenticated
    material for the strict full-chain guarantee and must not be upgraded
    to it silently.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("format_version") != STATEMENT_VERSION:
        return False
    if entry.get("schema") != STATEMENT_SCHEMA:
        return False
    if "statement_digest" not in entry or "statement_signature" not in entry:
        return False
    # confidence/input_commitment/output_commitment/policy_commitment are
    # nullable and may be entirely absent (canonical_json omits None
    # members), so only the always-present fields are required here.
    _nullable_fields = {"confidence", "input_commitment", "output_commitment", "policy_commitment"}
    required = [name for name in _STATEMENT_FIELDS if name not in _nullable_fields]
    return all(name in entry for name in required)


def _extract_statement(entry: dict[str, Any]) -> dict[str, Any]:
    """Pull the authenticated statement fields back out of a chain entry.

    Uses the *presented* values under the exact keys a caller would read to
    display the verdict (``entry["action"]``, not a separate copy), so a
    mutation of any bound field is, by construction, a mutation of what
    gets re-hashed here. Required fields use direct indexing (a missing key
    is a malformed entry, not a null statement field); the nullable
    commitment/confidence fields use ``.get`` since ``canonical_json`` omits
    ``None`` members and an entry legitimately signed without them will not
    carry the key at all.
    """
    return {
        "schema": entry["schema"],
        "format_version": entry["format_version"],
        "signer_id": entry["signer_id"],
        "execution_id": entry["execution_id"],
        "call_id": entry["call_id"],
        "action": entry["action"],
        "signals": entry["signals"],
        "confidence": entry.get("confidence"),
        "timestamp": entry["timestamp"],
        "sequence": entry["sequence"],
        "prev_link": entry["prev_link"],
        "input_commitment": entry.get("input_commitment"),
        "output_commitment": entry.get("output_commitment"),
        "policy_commitment": entry.get("policy_commitment"),
    }


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


@dataclass
class AepPrivateKey:
    """Ed25519 private key wrapper."""

    _key: Any  # Ed25519PrivateKey

    @classmethod
    def generate(cls) -> AepPrivateKey:
        _require_crypto()
        return cls(_key=Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: str | Path) -> AepPrivateKey:
        _require_crypto()
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        data = Path(path).read_bytes()
        key = load_pem_private_key(data, password=None)
        return cls(_key=key)

    def save(self, path: str | Path) -> None:
        import stat

        p = Path(path)
        pem = self._key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        p.write_bytes(pem)
        p.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner read/write only

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)

    @property
    def public_key(self) -> AepPublicKey:
        return AepPublicKey(_key=self._key.public_key())


@dataclass
class AepPublicKey:
    """Ed25519 public key wrapper."""

    _key: Any  # Ed25519PublicKey

    @classmethod
    def load(cls, path: str | Path) -> AepPublicKey:
        _require_crypto()
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        data = Path(path).read_bytes()
        key = load_pem_public_key(data)
        return cls(_key=key)

    def save(self, path: str | Path) -> None:
        pem = self._key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        Path(path).write_bytes(pem)

    def verify(self, signature: bytes, data: bytes) -> bool:
        try:
            self._key.verify(signature, data)
            return True
        except Exception:
            return False

    def to_base64(self) -> str:
        import base64

        raw = self._key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")


def generate_keypair() -> tuple[AepPrivateKey, AepPublicKey]:
    """Generate a new Ed25519 keypair for verdict signing."""
    private = AepPrivateKey.generate()
    return private, private.public_key


# ---------------------------------------------------------------------------
# Attestation engine (signing + Merkle chain)
# ---------------------------------------------------------------------------


@dataclass
class AttestationEngine:
    """Signs verdicts and maintains a Merkle audit chain.

    Each verdict is:
    1. Hashed (canonical JSON → SHA-256)
    2. Signed (Ed25519)
    3. Chained (SHA-256 of prev_chain_hash || verdict_hash)
    """

    _private_key: AepPrivateKey
    signer_id: str = "proxy:default"
    _chain_height: int = 0
    _prev_chain_hash: str = ""
    _chain: list[dict[str, Any]] = field(default_factory=list)
    # Distinct per engine instance (one per proxy process/session by
    # default). Bound into every statement this engine signs, so a valid
    # entry from one session can never be presented as part of another
    # session's chain (see #9657).
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _prev_statement_digest: str | None = field(default=None, init=False, repr=False)

    def sign_verdict(
        self,
        *,
        call_id: str,
        action: str,
        signals: list[dict[str, Any]],
        confidence: float | None = None,
        input_commitment: str | None = None,
        output_commitment: str | None = None,
        policy_commitment: str | None = None,
    ) -> dict[str, str]:
        """Sign a verdict and append it to the audit chain.

        Produces both the legacy narrow verdict signature (format_version 1
        fields — ``verdict_hash``/``signature``/``chain_hash``, unchanged
        computation, preserved for callers relying on
        :func:`verify_verdict`'s narrow meaning) and the format_version 2
        authenticated statement (``statement_digest``/``statement_signature``)
        that :func:`verify_chain` actually checks.

        ``input_commitment``/``output_commitment``/``policy_commitment`` are
        optional caller-supplied references (e.g. content hashes) bound into
        the signed statement when supplied. Their absence means "unbound",
        never "content/policy was checked" — which commitments are
        mandatory for a given profile is defined separately (#9652 S4).

        Returns a dict of HTTP header name → value pairs.
        """
        timestamp = datetime.now(UTC).isoformat()

        # --- format_version 1: narrow verdict hash + signature (unchanged) ---
        v_hash = verdict_hash(
            call_id=call_id,
            action=action,
            signals=signals,
            timestamp=timestamp,
            confidence=confidence,
        )
        hash_bytes = bytes.fromhex(v_hash.removeprefix("sha256:"))
        sig_bytes = self._private_key.sign(hash_bytes)
        signature = f"ed25519:{sig_bytes.hex()}"

        if self._chain_height == 0:
            chain_input = b"aep-genesis" + hash_bytes
        else:
            prev_bytes = bytes.fromhex(self._prev_chain_hash.removeprefix("sha256:"))
            chain_input = prev_bytes + hash_bytes
        chain_hash = f"sha256:{hashlib.sha256(chain_input).hexdigest()}"

        # --- format_version 2: full authenticated statement ---
        sequence = self._chain_height
        prev_link = (
            _genesis_link(self.execution_id)
            if self._prev_statement_digest is None
            else self._prev_statement_digest
        )
        statement = build_statement(
            signer_id=self.signer_id,
            execution_id=self.execution_id,
            call_id=call_id,
            action=action,
            signals=signals,
            timestamp=timestamp,
            sequence=sequence,
            prev_link=prev_link,
            confidence=confidence,
            input_commitment=input_commitment,
            output_commitment=output_commitment,
            policy_commitment=policy_commitment,
        )
        digest = statement_digest(statement)
        digest_bytes = bytes.fromhex(digest.removeprefix("sha256:"))
        statement_sig_bytes = self._private_key.sign(digest_bytes)
        statement_signature = f"ed25519:{statement_sig_bytes.hex()}"

        # --- record ---
        entry = {
            # legacy fields (format_version 1, unchanged meaning)
            "call_id": call_id,
            "action": action,
            "timestamp": timestamp,
            "verdict_hash": v_hash,
            "signature": signature,
            "chain_height": self._chain_height,
            "chain_hash": chain_hash,
            "prev_chain_hash": self._prev_chain_hash,
            # authenticated statement (format_version 2)
            **statement,
            "statement_digest": digest,
            "statement_signature": statement_signature,
        }
        self._chain.append(entry)
        self._prev_chain_hash = chain_hash
        self._prev_statement_digest = digest
        self._chain_height += 1

        # --- headers ---
        # Includes every field bound into the statement (timestamp, signals,
        # confidence included) so an HTTP-only consumer that archives
        # response headers — not just an in-process reader of
        # ``AttestationEngine.chain`` — can still reconstruct the complete
        # statement and recompute its digest. Omitting any of these from the
        # transport is exactly the original #9657 gap.
        headers = {
            "X-AEP-Verdict-Hash": v_hash,
            "X-AEP-Signature": signature,
            "X-AEP-Signer-Id": self.signer_id,
            "X-AEP-Chain-Height": str(entry["chain_height"]),
            "X-AEP-Chain-Hash": chain_hash,
            "X-AEP-Format-Version": str(STATEMENT_VERSION),
            "X-AEP-Execution-Id": self.execution_id,
            "X-AEP-Sequence": str(sequence),
            "X-AEP-Timestamp": timestamp,
            "X-AEP-Signals": json.dumps(signals, separators=(",", ":"), ensure_ascii=True),
            "X-AEP-Prev-Link": prev_link,
            "X-AEP-Statement-Digest": digest,
            "X-AEP-Statement-Signature": statement_signature,
        }
        if confidence is not None:
            headers["X-AEP-Confidence"] = json.dumps(confidence)
        return headers

    @property
    def chain(self) -> list[dict[str, Any]]:
        """Full audit chain for verification."""
        return list(self._chain)

    @property
    def chain_height(self) -> int:
        return self._chain_height


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_verdict(verdict_hash_str: str, signature_str: str, public_key: AepPublicKey) -> bool:
    """Verify a single digest/signature pair.

    Narrow, documented meaning: proves only that the holder of
    ``public_key``'s private key signed exactly these bytes. Says nothing
    about sequence position, session identity, or completeness — see
    :func:`verify_chain` / :func:`verify_chain_report` for that.

    Never raises: malformed hex or an unrelated exception from the
    underlying crypto library is treated as "does not verify".
    """
    try:
        hash_bytes = bytes.fromhex(verdict_hash_str.removeprefix("sha256:"))
        sig_bytes = bytes.fromhex(signature_str.removeprefix("ed25519:"))
    except (ValueError, AttributeError, TypeError):
        return False
    return public_key.verify(sig_bytes, hash_bytes)


@dataclass(frozen=True)
class ChainVerificationResult:
    """Detailed result of verifying a presented chain.

    ``status``:
    - ``"empty"``: no entries were presented (trivially valid).
    - ``"valid"``: every entry's digest, signature, and chain relationship
      (sequence position + authenticated previous link) verified, and all
      entries share one execution/session identity.
    - ``"legacy_unsupported"``: one or more entries lack the format_version
      2 authenticated statement fields. They may still satisfy the narrow
      ``verify_verdict`` primitive individually, but this presentation does
      not carry enough authenticated material for the strict full-chain
      guarantee. This is distinct from ``"invalid"`` — it is not evidence
      of tampering, only that the older format's bytes cannot honestly be
      reinterpreted under the stronger guarantee.
    - ``"invalid"``: tampering, a broken chain link, a cross-session splice,
      or a malformed entry was detected.

    A valid result proves the presented entries are an unbroken,
    authenticated prefix of one signer's one session. It does **not** prove
    that prefix is the complete history — detecting truncation requires a
    previously trusted checkpoint external to the chain itself (see
    docs/protocol/aep-attestation.md §6.5).
    """

    valid: bool
    status: str
    reason: str | None = None
    failed_index: int | None = None


def verify_chain(
    chain: list[dict[str, Any]],
    public_key: AepPublicKey,
    *,
    expected_execution_id: str | None = None,
) -> bool:
    """Verify an entire audit chain. See :func:`verify_chain_report` for detail.

    For format_version 2 entries (the default produced by
    :meth:`AttestationEngine.sign_verdict`), this recomputes and verifies:

    1. Every entry's full authenticated statement digest and signature
       (not a presented/unsigned hash — see #9657).
    2. Sequence positions are contiguous starting at 0 (no omission,
       no reordering).
    3. Each entry's ``prev_link`` equals the *previous entry's recomputed
       and verified* statement digest (not a value the presenter can
       recompute without the signing key).
    4. All entries share a single ``execution_id`` (no cross-session
       splicing), optionally pinned to ``expected_execution_id``.

    Chains containing any format_version 1 ("legacy") entries fail this
    strict check — see :class:`ChainVerificationResult` — rather than being
    silently upgraded to a guarantee their bytes never authenticated.
    """
    return verify_chain_report(chain, public_key, expected_execution_id=expected_execution_id).valid


def verify_chain_report(
    chain: list[dict[str, Any]],
    public_key: AepPublicKey,
    *,
    expected_execution_id: str | None = None,
) -> ChainVerificationResult:
    """Verify a chain and return a detailed, never-raising result.

    See :class:`ChainVerificationResult` for the meaning of each status.
    """
    if not chain:
        return ChainVerificationResult(valid=True, status="empty")

    for i, entry in enumerate(chain):
        if not _is_v2_entry(entry):
            log.warning("Chain entry %d lacks format_version 2 statement fields", i)
            return ChainVerificationResult(
                valid=False,
                status="legacy_unsupported",
                reason=(
                    "entry does not carry a complete format_version 2 authenticated "
                    "statement; legacy entries cannot be upgraded to the strict "
                    "full-chain guarantee"
                ),
                failed_index=i,
            )

    exec_ids = {entry.get("execution_id") for entry in chain}
    if len(exec_ids) != 1:
        log.warning("Chain mixes execution/session identities: %r", exec_ids)
        return ChainVerificationResult(
            valid=False,
            status="invalid",
            reason="chain mixes multiple execution/session identities (cross-session splice)",
        )
    execution_id = next(iter(exec_ids))
    if not isinstance(execution_id, str) or not execution_id:
        return ChainVerificationResult(
            valid=False,
            status="invalid",
            reason="execution_id must be a non-empty string",
        )
    if expected_execution_id is not None and execution_id != expected_execution_id:
        return ChainVerificationResult(
            valid=False,
            status="invalid",
            reason=(
                f"chain execution_id {execution_id!r} does not match "
                f"expected {expected_execution_id!r}"
            ),
        )

    prev_digest: str | None = None
    for i, entry in enumerate(chain):
        try:
            sequence = entry["sequence"]
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != i:
                log.warning("Sequence mismatch at position %d: got %r", i, sequence)
                return ChainVerificationResult(
                    valid=False,
                    status="invalid",
                    reason=f"sequence mismatch at position {i} (got {sequence!r})",
                    failed_index=i,
                )

            expected_prev_link = (
                _genesis_link(execution_id) if i == 0 else prev_digest
            )
            if entry.get("prev_link") != expected_prev_link:
                log.warning("Broken chain link at position %d", i)
                return ChainVerificationResult(
                    valid=False,
                    status="invalid",
                    reason=f"broken chain link at position {i}",
                    failed_index=i,
                )

            statement = _extract_statement(entry)
            digest = statement_digest(statement)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            log.warning("Malformed chain entry at position %d: %s", i, exc)
            return ChainVerificationResult(
                valid=False,
                status="invalid",
                reason=f"malformed entry at position {i}: {exc}",
                failed_index=i,
            )

        # The crypto decision is made against the *recomputed* digest, never
        # against a presented "statement_digest" field — trusting a
        # presented digest/hash without recomputing it from the raw fields
        # is exactly the #9657 bug.
        signature = entry.get("statement_signature", "")
        if not isinstance(signature, str) or not verify_verdict(digest, signature, public_key):
            log.warning("Invalid statement signature at position %d", i)
            return ChainVerificationResult(
                valid=False,
                status="invalid",
                reason=f"signature invalid at position {i}",
                failed_index=i,
            )

        prev_digest = digest

    return ChainVerificationResult(valid=True, status="valid")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "STATEMENT_SCHEMA",
    "STATEMENT_VERSION",
    "AepPrivateKey",
    "AepPublicKey",
    "AttestationEngine",
    "ChainVerificationResult",
    "build_statement",
    "canonical_json",
    "generate_keypair",
    "statement_digest",
    "verdict_hash",
    "verify_chain",
    "verify_chain_report",
    "verify_verdict",
]
