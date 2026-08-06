"""The FrameSource seam.

Every consumer of imagery talks to this protocol, never to a camera directly.
That is what makes `FileReplaySource` possible, and replay is what makes the
whole system testable without sleeping (design.md §17, §21): a detector change
can be re-run over the full corpus of recorded nights in minutes.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from ..types import Frame


class FrameSourceError(RuntimeError):
    """Raised when a source cannot be opened or configured as required."""


@runtime_checkable
class FrameSource(Protocol):
    """A timestamped image stream.

    `read()` returns None to signal a *transient* failure the caller may retry;
    exhaustion of a finite source is signalled by `exhausted` becoming True.
    Conflating the two would make a replay run look like a broken camera.
    """

    def open(self) -> None: ...

    def read(self) -> Optional[Frame]: ...

    def close(self) -> None: ...

    @property
    def exhausted(self) -> bool: ...

    def device_profile(self) -> dict[str, Any]: ...
