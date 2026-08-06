"""Sensor-timed cueing (G9), and the lock that keeps it disabled.

This module is the payoff of the whole camera branch and the thing most likely
to produce a wrong answer, so it is built to refuse to run.

`SensorTimingAuthorization` reads `validation_results` and reports whether H1 has
passed. `CueController` will not accept a sensor-timing configuration without a
passing record — it raises at construction, before a night starts, rather than
degrading quietly at 04:00. There is no flag to override it. Adding one would
defeat the purpose, since the person who would use it is the person the lock
protects.

The reasoning: an eye-movement index that has not been checked against a
reference is an unvalidated number. Letting it decide when to make a noise at a
sleeping person would produce cue timings that feel principled and are not, and
the resulting data would look like evidence about cue timing while actually
being evidence about nothing (design.md §8, §22).

The HMM here exists because per-second estimates are far noisier than the thing
they estimate. Sleep states persist for tens of minutes, so a two-state model
with sticky transitions recovers most of the achievable gain for a fraction of
the complexity of anything learned end-to-end.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("morpheus.sensor_timing")


class SensorTimingLocked(RuntimeError):
    """Raised when sensor-timed cueing is requested without validation."""


@dataclass(frozen=True)
class SensorTimingAuthorization:
    authorized: bool
    reason: str
    auc: Optional[float] = None
    validated_at: Optional[str] = None

    @classmethod
    def from_db(cls, conn: sqlite3.Connection) -> "SensorTimingAuthorization":
        from ..reference.validate import AUC_PASS, latest_passing

        try:
            row = latest_passing(conn)
        except sqlite3.OperationalError:
            return cls(False, "validation_results table missing; run migrations")

        if row is None:
            return cls(
                False,
                f"no passing H1 validation on record. Sensor-timed cueing requires a "
                f"validated eye-movement index (AUC >= {AUC_PASS} against a reference "
                f"device, evaluated on held-out nights). Run `morpheus validate` first.",
            )
        return cls(
            True,
            f"validated at AUC {row['auc']:.3f} on {row['created_at']}",
            auc=float(row["auc"]),
            validated_at=row["created_at"],
        )

    @classmethod
    def unlocked_for_testing(cls, auc: float = 0.75) -> "SensorTimingAuthorization":
        """Explicit test-only constructor.

        Named so that any production use of it is obvious in review. Tests need
        to exercise the enabled path; nothing else should construct this.
        """
        return cls(True, f"TEST AUTHORIZATION (auc={auc})", auc=auc)


@dataclass
class SensorTimingConfig:
    """Parameters for G9. Only usable alongside a passing authorization."""

    # Personalised threshold on the smoothed eye-activity index, expressed as a
    # robust z-score against the night's own baseline rather than an absolute,
    # since the index has no meaningful units and varies by setup.
    activity_z_threshold: float = 1.5
    min_persistence_s: float = 20.0
    # Refuse to act on the index when coverage is poor. A smoothed value built
    # from three visible seconds out of thirty is not an estimate.
    min_coverage: float = 0.5
    use_hmm: bool = True
    hmm_stickiness: float = 0.97


class StickyTwoStateHMM:
    """Two-state Gaussian HMM with a fixed, sticky transition matrix.

    Not fitted end-to-end. The transition probability encodes a fact already
    known — sleep states last tens of minutes, so second-to-second switching is
    implausible — and fixing it avoids burning scarce data on re-learning it.
    Emissions are Gaussian, fitted per night by splitting at the median, which is
    crude but self-calibrating across setups where absolute levels differ.

    Returns posterior probability of the high-activity state, which is a
    smoothed index and explicitly not a claim about sleep stage.
    """

    def __init__(self, stickiness: float = 0.97) -> None:
        if not 0.5 < stickiness < 1.0:
            raise ValueError("stickiness must be in (0.5, 1)")
        stay = stickiness
        self.transitions = np.array([[stay, 1 - stay], [1 - stay, stay]])

    def posterior(self, x: Sequence[float]) -> np.ndarray:
        values = np.asarray(list(x), dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 10:
            return np.full(values.size, np.nan)

        observed = values[finite]
        split = np.median(observed)
        low, high = observed[observed <= split], observed[observed > split]
        if low.size < 2 or high.size < 2:
            return np.full(values.size, np.nan)

        means = np.array([low.mean(), high.mean()])
        # Floor the variance: a night where the index barely moves would
        # otherwise produce near-zero sigma and infinitely confident emissions.
        sigmas = np.array([max(low.std(), 1e-6), max(high.std(), 1e-6)])
        if means[1] - means[0] < 1e-9:
            return np.full(values.size, np.nan)

        emissions = np.exp(
            -0.5 * ((observed[:, None] - means[None, :]) / sigmas[None, :]) ** 2
        ) / sigmas[None, :]
        emissions = np.clip(emissions, 1e-300, None)

        n = observed.size
        alpha = np.zeros((n, 2))
        scale = np.zeros(n)
        alpha[0] = np.array([0.5, 0.5]) * emissions[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.transitions) * emissions[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)

        beta = np.zeros((n, 2))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = self.transitions @ (emissions[t + 1] * beta[t + 1])
            beta[t] /= max(beta[t].sum(), 1e-300)

        gamma = alpha * beta
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

        out = np.full(values.size, np.nan)
        out[finite] = gamma[:, 1]
        return out


@dataclass
class ActivityIndex:
    """Rolling eye-activity index with a personalised threshold.

    Thresholds come from the night's own distribution using median and MAD
    rather than mean and standard deviation. Sleep features are heavy-tailed,
    and a handful of large bursts would drag a mean-based threshold above the
    very events it is meant to catch.
    """

    config: SensorTimingConfig
    _history: list[float] = None  # type: ignore[assignment]
    _above_since: Optional[float] = None

    def __post_init__(self) -> None:
        self._history = []

    def update(self, value: Optional[float], coverage: float, t_mono: float) -> bool:
        """Feed one second. Returns True when a sustained burst is active."""
        if value is None or not np.isfinite(value) or coverage < self.config.min_coverage:
            self._above_since = None
            return False

        self._history.append(float(value))
        if len(self._history) > 3600:
            self._history = self._history[-3600:]
        if len(self._history) < 120:
            return False  # not enough baseline to threshold against

        arr = np.asarray(self._history)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        # 1.4826 rescales MAD to be comparable with a standard deviation.
        scale = max(mad * 1.4826, 1e-9)
        z = (value - median) / scale

        if z >= self.config.activity_z_threshold:
            if self._above_since is None:
                self._above_since = t_mono
            return (t_mono - self._above_since) >= self.config.min_persistence_s
        self._above_since = None
        return False

    def reset(self) -> None:
        self._history = []
        self._above_since = None
