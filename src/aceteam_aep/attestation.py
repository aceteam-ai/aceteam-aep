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

    Fields sorted alphabetically, minimal encoding, null fields omitted.
    """

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in sorted(obj.items()) if v is not None}
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
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
        pem = self._key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        Path(path).write_bytes(pem)

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

    def sign_verdict(
        self,
        *,
        call_id: str,
        action: str,
        signals: list[dict[str, Any]],
        confidence: float | None = None,
    ) -> dict[str, str]:
        """Sign a verdict and append to the Merkle chain.

        Returns a dict of HTTP header name → value pairs.
        """
        timestamp = datetime.now(UTC).isoformat()

        # 1. Verdict hash
        v_hash = verdict_hash(
            call_id=call_id,
            action=action,
            signals=signals,
            timestamp=timestamp,
            confidence=confidence,
        )

        # 2. Sign
        hash_bytes = bytes.fromhex(v_hash.removeprefix("sha256:"))
        sig_bytes = self._private_key.sign(hash_bytes)
        signature = f"ed25519:{sig_bytes.hex()}"

        # 3. Merkle chain
        if self._chain_height == 0:
            chain_input = b"aep-genesis" + hash_bytes
        else:
            prev_bytes = bytes.fromhex(self._prev_chain_hash.removeprefix("sha256:"))
            chain_input = prev_bytes + hash_bytes

        chain_hash = f"sha256:{hashlib.sha256(chain_input).hexdigest()}"

        # 4. Record
        entry = {
            "call_id": call_id,
            "action": action,
            "timestamp": timestamp,
            "verdict_hash": v_hash,
            "signature": signature,
            "chain_height": self._chain_height,
            "chain_hash": chain_hash,
            "prev_chain_hash": self._prev_chain_hash,
        }
        self._chain.append(entry)
        self._prev_chain_hash = chain_hash
        self._chain_height += 1

        # 5. Return headers
        return {
            "X-AEP-Verdict-Hash": v_hash,
            "X-AEP-Signature": signature,
            "X-AEP-Signer-Id": self.signer_id,
            "X-AEP-Chain-Height": str(entry["chain_height"]),
            "X-AEP-Chain-Hash": chain_hash,
        }

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
    """Verify a single verdict signature."""
    hash_bytes = bytes.fromhex(verdict_hash_str.removeprefix("sha256:"))
    sig_bytes = bytes.fromhex(signature_str.removeprefix("ed25519:"))
    return public_key.verify(sig_bytes, hash_bytes)


def verify_chain(chain: list[dict[str, Any]], public_key: AepPublicKey) -> bool:
    """Verify an entire Merkle audit chain.

    Checks:
    1. Every signature is valid
    2. Chain hashes are correctly computed
    3. Heights are monotonically increasing
    """
    prev_chain_hash = ""

    for i, entry in enumerate(chain):
        # Check height
        if entry["chain_height"] != i:
            log.warning(
                "Chain height mismatch at index %d: expected %d, got %d",
                i,
                i,
                entry["chain_height"],
            )
            return False

        # Verify signature
        if not verify_verdict(entry["verdict_hash"], entry["signature"], public_key):
            log.warning("Invalid signature at chain height %d", i)
            return False

        # Verify chain hash
        v_hash_bytes = bytes.fromhex(entry["verdict_hash"].removeprefix("sha256:"))
        if i == 0:
            expected_input = b"aep-genesis" + v_hash_bytes
        else:
            prev_bytes = bytes.fromhex(prev_chain_hash.removeprefix("sha256:"))
            expected_input = prev_bytes + v_hash_bytes

        expected_chain_hash = f"sha256:{hashlib.sha256(expected_input).hexdigest()}"
        if entry["chain_hash"] != expected_chain_hash:
            log.warning("Chain hash mismatch at height %d", i)
            return False

        prev_chain_hash = entry["chain_hash"]

    return True


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AepPrivateKey",
    "AepPublicKey",
    "AttestationEngine",
    "canonical_json",
    "generate_keypair",
    "verdict_hash",
    "verify_chain",
    "verify_verdict",
]
