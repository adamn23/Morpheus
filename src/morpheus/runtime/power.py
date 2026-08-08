"""Keeping macOS awake for the duration of a run.

A laptop that suspends at 02:00 does not produce a short night; it produces a
night with an invisible hole in it, because `time.monotonic()` on macOS does not
advance during system sleep. The recorder would resume and carry on numbering
seconds as though nothing happened. Preventing the sleep is the first defence;
detecting the gap anyway (see `runtime.recorder`) is the second, because the
assertion can be lost to a lid close or a power event.

`caffeinate` is used rather than an IOPMAssertion via ctypes: it is a supported
system binary, it dies with its child, and it needs no framework bindings. The
cost is one subprocess.

**The display must be allowed to sleep.** `-d` holds the screen on and `-u`
actively turns it on, and either one puts a lit screen next to the pillow for
eight hours. This project's primary risk is degrading the sleep it is measuring,
so a light source at the bedside is both a harm and an uncontrolled confound
across every night of the study. Keeping the system awake and the display dark
are separate assertions, and only the first is wanted here.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from types import TracebackType
from typing import Optional


class SleepPreventer:
    """Context manager holding a system sleep assertion for its lifetime.

    A no-op on non-macOS platforms and when `caffeinate` is unavailable; callers
    should check `active` and warn rather than assume it worked.
    """

    def __init__(self, *, enabled: bool = True, reason: str = "Morpheus recording") -> None:
        self._enabled = enabled
        self._reason = reason
        self._proc: Optional[subprocess.Popen] = None
        self._status = "not started"

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> None:
        if not self._enabled:
            self._status = "disabled by configuration"
            return
        if platform.system() != "Darwin":
            self._status = f"unsupported platform ({platform.system()}); no assertion held"
            return
        binary = shutil.which("caffeinate")
        if binary is None:
            self._status = "caffeinate not found on PATH; system may sleep mid-run"
            return
        try:
            self._proc = subprocess.Popen(
                # -i idle, -m disk, -s system-on-AC. Deliberately no -d/-u:
                # see the module docstring. The screen must go dark.
                [binary, "-ims"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._status = f"could not start caffeinate: {exc}"
            return
        self._status = f"holding sleep assertion (caffeinate pid {self._proc.pid})"

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._status = "released"

    def __enter__(self) -> "SleepPreventer":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.stop()
