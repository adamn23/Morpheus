"""H1 validation: does a camera-derived index track REM?

This module decides whether the eye-movement branch lives or dies, so its
methodology matters more than its code.

**Held-out nights, never held-out windows.** Consecutive 30-second epochs are
massively autocorrelated — sleep states persist for tens of minutes. Splitting
randomly across windows lets a model memorise a night's idiosyncrasies and score
an AUC near 1.0 while being useless on a night it has not seen. This is the
single easiest way to fool yourself in sleep classification, and it is why
`GroupKFold` groups by session throughout.

**A pre-committed threshold.** AUC >= 0.70 proceeds, < 0.65 kills the branch
(design.md §22). Fixed in the design document before any data existed,
imported here as a constant rather than passed in, so the go/no-go cannot be
renegotiated by an analyst who has already seen the number.

**A recorded verdict.** The result is written to `validation_results`, and G9
reads that table. Failing validation is not advisory — the sensor-timed cueing
path is inert without a passing row.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

# Pre-committed in design.md §22. Do not tune.
AUC_PASS = 0.70
AUC_KILL = 0.65
MIN_NIGHTS = 5
MIN_REM_EPOCHS = 40

# Features available from M0/M1. Ordered for stable coefficient reporting.
FEATURE_COLUMNS = (
    "eye_flow_l", "eye_flow_r", "eye_flow_bilateral_corr",
    "lid_disp_l", "lid_disp_r",
    "global_motion", "bed_motion", "face_motion",
    "head_motion", "resp_proxy",
)


@dataclass
class ValidationResult:
    hypothesis: str
    reference_source: str
    nights_used: int
    held_out_nights: int
    n_epochs: int
    n_rem: int
    auc: Optional[float]
    auc_ci: tuple[Optional[float], Optional[float]]
    brier: Optional[float]
    calibration: list[tuple[float, float]] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    insufficient: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.auc is not None and self.auc >= AUC_PASS

    @property
    def killed(self) -> bool:
        return self.auc is not None and self.auc < AUC_KILL

    @property
    def verdict(self) -> str:
        if self.insufficient:
            return "INSUFFICIENT DATA"
        if self.passed:
            return "PASS"
        if self.killed:
            return "FAIL"
        return "INCONCLUSIVE"


def _bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, groups: np.ndarray, *, draws: int = 1000, seed: int = 7
) -> tuple[Optional[float], Optional[float]]:
    """Bootstrap over *nights*, not epochs.

    Resampling epochs would treat 900 correlated samples from one night as 900
    independent ones and produce an interval far too narrow to be honest.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if unique.size < 3:
        return (None, None)

    values: list[float] = []
    for _ in range(draws):
        picked = rng.choice(unique, size=unique.size, replace=True)
        mask = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        if np.unique(y[mask]).size < 2:
            continue
        values.append(float(roc_auc_score(y[mask], scores[mask])))
    if len(values) < 50:
        return (None, None)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def validate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
    hypothesis: str = "H1: camera-derived index discriminates REM from non-REM",
    reference_source: str = "unknown",
) -> ValidationResult:
    """Cross-validated, night-grouped evaluation of a REM discriminator."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_nights = int(np.unique(groups).size)
    n_rem = int(labels.sum())

    base = ValidationResult(
        hypothesis=hypothesis,
        reference_source=reference_source,
        nights_used=n_nights,
        held_out_nights=0,
        n_epochs=int(labels.size),
        n_rem=n_rem,
        auc=None,
        auc_ci=(None, None),
        brier=None,
        features_used=list(feature_names),
    )

    if n_nights < MIN_NIGHTS:
        base.insufficient = f"only {n_nights} nights; need at least {MIN_NIGHTS}"
        return base
    if n_rem < MIN_REM_EPOCHS:
        base.insufficient = f"only {n_rem} REM epochs; need at least {MIN_REM_EPOCHS}"
        return base
    if np.unique(labels).size < 2:
        base.insufficient = "labels contain only one class"
        return base

    n_splits = min(5, n_nights)
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.zeros_like(labels, dtype=float)

    for train_idx, test_idx in splitter.split(features, labels, groups):
        if np.unique(labels[train_idx]).size < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(features[train_idx], labels[train_idx])
        predictions[test_idx] = model.predict_proba(features[test_idx])[:, 1]

    auc = float(roc_auc_score(labels, predictions))
    brier = float(brier_score_loss(labels, predictions))
    ci = _bootstrap_auc_ci(labels, predictions, groups)

    # Fit once on everything purely to report coefficient direction. This model
    # is never evaluated — the AUC above comes from held-out nights only.
    full = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    full.fit(features, labels)
    coefs = full[-1].coef_[0]

    base.held_out_nights = n_splits
    base.auc = auc
    base.auc_ci = ci
    base.brier = brier
    base.coefficients = {
        name: float(value) for name, value in zip(feature_names, coefs)
    }
    base.calibration = _calibration_curve(labels, predictions)
    return base


def _calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[tuple[float, float]]:
    """Predicted vs observed frequency. A high AUC with bad calibration means
    the ranking is useful but the probabilities are not, which matters if a
    threshold is ever used to gate a cue."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[tuple[float, float]] = []
    for low, high in zip(edges, edges[1:]):
        mask = (p >= low) & (p < high)
        if mask.sum() >= 5:
            out.append((float(p[mask].mean()), float(y[mask].mean())))
    return out


def record(conn: sqlite3.Connection, result: ValidationResult, notes: str = "") -> int:
    """Persist the verdict. This row is what G9 checks before it will activate."""
    cur = conn.execute(
        "INSERT INTO validation_results (created_at, hypothesis, reference_source, "
        "nights_used, held_out_nights, auc, auc_ci_low, auc_ci_high, brier, threshold, "
        "passed, model_version, metrics_json, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            result.hypothesis, result.reference_source,
            result.nights_used, result.held_out_nights,
            result.auc, result.auc_ci[0], result.auc_ci[1], result.brier,
            AUC_PASS, int(result.passed), "logreg-v1",
            json.dumps({
                "coefficients": result.coefficients,
                "calibration": result.calibration,
                "features": result.features_used,
                "n_epochs": result.n_epochs,
                "n_rem": result.n_rem,
                "insufficient": result.insufficient,
            }),
            notes,
        ),
    )
    return int(cur.lastrowid)


def latest_passing(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """The record G9 requires. None means sensor-timed cueing stays disabled."""
    return conn.execute(
        "SELECT * FROM validation_results WHERE passed = 1 AND auc >= ? "
        "ORDER BY created_at DESC LIMIT 1",
        (AUC_PASS,),
    ).fetchone()


def format_result(result: ValidationResult) -> str:
    lines: list[str] = []
    add = lines.append

    add("H1 validation — camera index vs reference REM")
    add("=" * 70)
    add(f"  reference        {result.reference_source}")
    add(f"  nights           {result.nights_used} ({result.held_out_nights}-fold, grouped by night)")
    add(f"  epochs           {result.n_epochs:,} ({result.n_rem:,} REM)")
    add("")

    if result.insufficient:
        add(f"  INSUFFICIENT DATA: {result.insufficient}")
        add("")
        add("  No verdict is recorded. G9 stays disabled, which is the correct")
        add("  behaviour — an unvalidated detector must not influence cue timing.")
        return "\n".join(lines)

    ci_low, ci_high = result.auc_ci
    ci_text = f" [{ci_low:.3f}, {ci_high:.3f}]" if ci_low is not None else " (CI needs >=3 nights)"
    add(f"  AUC              {result.auc:.3f}{ci_text}")
    add(f"  Brier score      {result.brier:.4f}")
    add(f"  thresholds       pass >= {AUC_PASS}, kill < {AUC_KILL}")
    add("")

    if result.coefficients:
        add("Coefficient direction (standardised, full-data fit, not evaluated)")
        add("-" * 70)
        ranked = sorted(result.coefficients.items(), key=lambda kv: -abs(kv[1]))
        for name, value in ranked[:8]:
            add(f"  {name:<26} {value:+.3f}")
        add("")

    add(f"VERDICT: {result.verdict}")
    add("=" * 70)
    if result.passed:
        add("  The index discriminates REM better than the pre-committed threshold.")
        add("  G9 may now be enabled, and must then be A/B'd against the scheduler")
        add("  before it is trusted to improve anything.")
    elif result.killed:
        add("  Below the abandon threshold. Per design.md §23, the eye-movement")
        add("  branch is finished: keep the camera as a motion guard, publish the")
        add("  negative result, and do not tune until it passes. Tuning against a")
        add("  threshold you have already seen is how noise becomes a finding.")
    else:
        add("  Between the thresholds. Collect more nights. Do not move the")
        add("  thresholds, and do not enable G9 in the meantime.")
    return "\n".join(lines)
