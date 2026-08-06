"""Sealing condition assignments, and auditing every unsealing.

An honest account of what this can and cannot do, because overstating it would
be worse than not having it.

Morpheus is a single-user system. The participant is also the developer, the
analyst, and the person holding the machine and any key on it. Against that
person, cryptography buys nothing: they can read this file. So this is
deliberately called sealing rather than encryption, and it is built to achieve
two things that *are* achievable:

  1. **Casual exposure is prevented.** Opening the database, grepping the data
     directory, or glancing at a table does not reveal tonight's arm. Most
     accidental unblinding is exactly this careless, and obfuscation stops it.
  2. **Deliberate exposure is recorded.** Every unsealing writes to
     `reveal_audit` with a reason and a legitimacy flag. Blinding that cannot be
     enforced can still be *measured*, and a study whose unblinding rate is
     known is analysable in a way that one relying on good intentions is not.

The third mechanism is procedural and lives in `assignments.py`: the reveal path
refuses to unseal a night until a morning report exists for it. Combined with
the `guessed_condition` field on every report, that gives an empirical handle on
how often blinding actually held (design.md §15.2).

If genuine blinding is ever needed, the honest route is a passphrase held by
someone else — not a better cipher on the participant's own disk.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
from pathlib import Path

log = logging.getLogger("morpheus.blinding")

KEY_FILENAME = ".blinding_key"


def load_or_create_key(data_dir: Path) -> bytes:
    """Read the sealing key, creating one on first use."""
    path = Path(data_dir) / KEY_FILENAME
    if path.exists():
        return path.read_bytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass  # best effort; the filesystem may not support it
    log.info("created sealing key at %s", path)
    return key


def _keystream(key: bytes, nonce: str, length: int) -> bytes:
    """HMAC-SHA256 in counter mode.

    Sufficient for the stated goal — making the value unreadable at a glance —
    and dependency-free. It is not authenticated encryption and is not
    presented as such.
    """
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hmac.new(key, f"{nonce}:{counter}".encode(), hashlib.sha256).digest())
        counter += 1
    return bytes(out[:length])


def seal(key: bytes, nonce: str, plaintext: str) -> str:
    data = plaintext.encode()
    stream = _keystream(key, nonce, len(data))
    return base64.urlsafe_b64encode(bytes(a ^ b for a, b in zip(data, stream))).decode()


def unseal(key: bytes, nonce: str, sealed: str) -> str:
    data = base64.urlsafe_b64decode(sealed.encode())
    stream = _keystream(key, nonce, len(data))
    return bytes(a ^ b for a, b in zip(data, stream)).decode()


class BlindingError(RuntimeError):
    """Raised when an unseal is attempted without the conditions being met."""
