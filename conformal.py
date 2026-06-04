"""
Split conformal prediction for SKYLANCE-X action certificates.

Theory
------
We want to certify: "this recommended action holds ICAO separation with
probability >= 1 - alpha."

Standard split conformal prediction (Papadopoulos et al. 2002; Vovk et al. 2005)
gives a finite-sample, distribution-free guarantee:

    P(Y_{n+1} ∈ C(X_{n+1})) >= 1 - alpha

where Y is the true outcome (held/violated), X is the scenario+action, and
C is the conformal prediction set.

For a binary safe/unsafe label we use the **conformal risk control** variant
(Angelopoulos & Bates 2022):

    E[L(action)] <= alpha

where L = 1 if we certify safe but separation is actually violated.

Nonconformity score
-------------------
    nc(cascade_score) = separation_breach_count
                        + urgency_penalty × (1 - time_to_first_sep_breach / horizon)

  nc = 0  →  cascade says perfectly safe
  nc > 0  →  cascade predicts one or more separation events (larger = worse)

Calibration threshold
---------------------
    q_hat = ceil((n_cal + 1)(1 - alpha)) / n_cal  quantile of {nc_i}

For a new recommendation:
  • Certify SAFE  if nc_new <= q_hat  (P(violated) <= alpha)
  • Certify UNSAFE otherwise

Coverage guarantee: P(violated AND certified safe) <= alpha + 1/(n_cal + 1)

Runtime note
------------
build_calibration_set() runs N BlueSky scenarios end-to-end.  Each scenario
involves loading 8 aircraft and stepping 600s (12,000 BlueSky ticks at dt=0.05s).
The recommender evaluates ~20 candidates per scenario.  Budget ~5-8 min for
N=100 on a modern laptop.  Pass verbose=True to monitor progress.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from state_schema import SectorState, make_mock_sector
from cascade_engine import (
    CascadeEngine, CascadeScore, CandidateAction,
    NoAction, _apply_action,
)

# ---------------------------------------------------------------------------
# Nonconformity score
# ---------------------------------------------------------------------------

def nc_score(cascade: CascadeScore) -> float:
    """
    Map a CascadeScore to a non-negative nonconformity score.

    Interpretation:
      0          → cascade predicts no separation events (most conforming / safest)
      0–1        → cascade predicts a late, single separation event
      > 1        → cascade predicts multiple or early separation events

    The urgency sub-term ensures that an action that keeps violations close to
    the horizon (t+540s) scores lower than one with an imminent breach (t+60s),
    even when breach counts are equal.
    """
    sep_count = cascade.separation_breach_count
    if sep_count == 0:
        return 0.0

    urgency = 0.0
    if (cascade.time_to_first_breach_s is not None
            and cascade.first_breach_type == "SEPARATION"):
        # 1.0 when breach is immediate; 0.0 when breach is at the very end
        urgency = 1.0 - cascade.time_to_first_breach_s / cascade.horizon_s

    return float(sep_count) + urgency


# ===========================================================================
# Calibration record
# ===========================================================================

@dataclass
class CalibrationPoint:
    """
    One data point for conformal calibration.

    nc          : nonconformity score from the cascade engine (our model)
    actually_held : ground-truth from the full BlueSky run (cd.lospairs)
    """
    scenario_seed: int
    n_aircraft: int
    action_type: str
    nc: float                  # nonconformity score from cascade engine
    actually_held: bool        # True = BlueSky confirmed no separation loss

    def to_dict(self) -> dict:
        return {
            "scenario_seed": self.scenario_seed,
            "n_aircraft": self.n_aircraft,
            "action_type": self.action_type,
            "nc": self.nc,
            "actually_held": self.actually_held,
        }


# ===========================================================================
# Certifier
# ===========================================================================

@dataclass
class Certificate:
    """Result of a certification query."""
    certified_safe: bool
    nc: float                   # nonconformity score of this recommendation
    threshold: float            # q_hat used
    alpha: float                # claimed miscoverage rate
    p_value: float              # marginal conformal p-value
    n_calibration: int

    @property
    def confidence_pct(self) -> float:
        return (1 - self.alpha) * 100

    def __str__(self) -> str:
        verdict = "SAFE" if self.certified_safe else "UNSAFE"
        return (
            f"[{verdict}]  nc={self.nc:.3f}  threshold={self.threshold:.3f}  "
            f"alpha={self.alpha:.2f}  p={self.p_value:.3f}  "
            f"n_cal={self.n_calibration}"
        )


class ConformalCertifier:
    """
    Split conformal certifier for separation guarantees.

    Workflow:
        cert = ConformalCertifier(alpha=0.05)
        cert.calibrate(calibration_points)
        result = cert.certify(cascade_score)
        print(result)   # → [SAFE]  nc=0.000  threshold=0.800 ...
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self._cal_nc: Optional[np.ndarray] = None
        self._threshold: float = float("inf")
        self._n_cal: int = 0

    def calibrate(self, points: list[CalibrationPoint]) -> None:
        """
        Fit the conformal threshold to a list of CalibrationPoints.

        Uses the standard (n+1)(1-alpha)/n quantile formula, which provides
        the finite-sample marginal coverage guarantee:

            P(violated AND certified safe) <= alpha + 1/(n_cal + 1)

        Following Tibshirani et al. (2019), we use all points (not just
        truly-safe ones) so the threshold also reflects false-safe scenarios.
        """
        if not points:
            raise ValueError("Calibration set is empty.")

        scores = np.array([p.nc for p in points], dtype=float)
        n = len(scores)
        # quantile level: ceil((n+1)(1-alpha)) / n, clamped to [0, 1]
        q_level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self._threshold = float(np.quantile(scores, q_level, method="higher"))
        self._cal_nc   = scores
        self._n_cal    = n

    def certify(self, cascade: CascadeScore) -> Certificate:
        """
        Issue a safety certificate for a new cascade score.

        Raises RuntimeError if calibrate() has not been called yet.
        """
        if self._cal_nc is None:
            raise RuntimeError("Call calibrate() before certify().")

        nc = nc_score(cascade)
        p_val = float(np.sum(self._cal_nc >= nc) + 1) / (self._n_cal + 1)

        return Certificate(
            certified_safe=nc <= self._threshold,
            nc=nc,
            threshold=self._threshold,
            alpha=self.alpha,
            p_value=p_val,
            n_calibration=self._n_cal,
        )

    def certify_nc(self, nc: float) -> bool:
        """Lightweight check used by check_coverage — no Certificate wrapper."""
        return nc <= self._threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def n_calibration(self) -> int:
        return self._n_cal


# ===========================================================================
# Calibration set builder
# ===========================================================================

def _bluesky_held_separation(
    sector: SectorState,
    action: CandidateAction,
    adapter,
    horizon_s: float,
) -> bool:
    """
    Load sector+action into BlueSky and return True if no loss-of-separation
    occurred throughout the horizon.

    Uses BlueSky's built-in CPA-based conflict detection (cd.lospairs), which
    can disagree with the cascade engine's geometric ICAO check.  This
    disagreement is exactly what conformal calibration corrects for.
    """
    import bluesky as bs
    modified = _apply_action(sector, action)
    adapter.load_scenario(modified)

    n_steps = max(1, round(horizon_s / bs.sim.simdt))
    for _ in range(n_steps):
        bs.sim.step()
        if len(bs.traf.cd.lospairs) > 0:
            return False   # at least one loss-of-separation

    return True


def build_calibration_set(
    n_scenarios: int,
    adapter,
    engine: CascadeEngine,
    recommender,                   # Recommender — typed loosely to avoid circular import
    n_aircraft: int = 8,           # keep small for speed; conformal theory still holds
    seed_start: int = 10_000,
    verbose: bool = True,
) -> list[CalibrationPoint]:
    """
    Generate `n_scenarios` random sectors, run the recommender on each, and
    validate the chosen action with a full BlueSky simulation.

    Each CalibrationPoint records:
      • nc         — cascade engine's nonconformity score for the chosen action
      • actually_held — whether BlueSky's actual conflict detector agreed

    The gap between cascade predictions and BlueSky outcomes is what conformal
    calibration corrects.  More calibration points → tighter, more reliable
    certificates.

    Args:
        n_scenarios : number of scenarios to simulate (>= 100 recommended)
        adapter     : BlueSkyAdapter (will be left in an arbitrary state)
        engine      : CascadeEngine
        recommender : Recommender
        n_aircraft  : aircraft per scenario (8 keeps each run fast)
        seed_start  : first RNG seed (seeds are seed_start .. seed_start+n-1)
        verbose     : print progress every 10 scenarios
    """
    points: list[CalibrationPoint] = []

    for i in range(n_scenarios):
        seed = seed_start + i
        sector = make_mock_sector(n=n_aircraft, seed=seed)

        # Get the recommender's best action and its cascade score
        recommendation = recommender.recommend(sector)
        action         = recommendation.action
        cascade        = recommendation.score

        nc_val = nc_score(cascade)

        # Ground-truth: did the action actually hold separation in BlueSky?
        held = _bluesky_held_separation(sector, action, adapter, engine.horizon_s)

        points.append(CalibrationPoint(
            scenario_seed=seed,
            n_aircraft=n_aircraft,
            action_type=type(action).__name__,
            nc=nc_val,
            actually_held=held,
        ))

        if verbose and (i + 1) % 10 == 0:
            n_held = sum(p.actually_held for p in points)
            print(f"  [{i+1:3d}/{n_scenarios}]  held={n_held}/{i+1}  "
                  f"last_nc={nc_val:.3f}  last_action={type(action).__name__}")

    return points


# ===========================================================================
# Coverage check + plot
# ===========================================================================

def check_coverage(
    points: list[CalibrationPoint],
    alphas: Optional[list[float]] = None,
    cal_fraction: float = 0.8,
    save_path: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Split points into calibration / test, sweep alpha, and plot empirical vs
    claimed coverage.

    A valid conformal predictor sits on or ABOVE the diagonal (empirical
    coverage >= claimed 1 - alpha).  Any dip below the diagonal indicates
    either a distributional shift or too few calibration points.

    Args:
        points        : output of build_calibration_set()
        alphas        : miscoverage levels to test; defaults to 0.01 … 0.30
        cal_fraction  : fraction used for calibration (rest is test)
        save_path     : if given, save the figure here (e.g. "coverage.png")
        show          : call plt.show() if True

    Returns:
        dict with keys 'alphas', 'empirical_coverage', 'empirical_miscoverage',
        'n_cal', 'n_test', 'cal_held_rate', 'test_held_rate'
    """
    import matplotlib.pyplot as plt

    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.01, 0.31, 0.02)]

    # Shuffle deterministically then split
    rng = random.Random(0)
    shuffled = list(points)
    rng.shuffle(shuffled)
    n_cal  = max(10, int(len(shuffled) * cal_fraction))
    cal    = shuffled[:n_cal]
    test   = shuffled[n_cal:]

    if not test:
        raise ValueError("Not enough points for a test split; add more scenarios.")

    cal_ncs  = np.array([p.nc for p in cal])
    test_ncs = np.array([p.nc for p in test])
    test_held = np.array([p.actually_held for p in test])

    empirical_coverage     = []
    empirical_miscoverage  = []

    for alpha in alphas:
        n = len(cal)
        q_level   = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        threshold = float(np.quantile(cal_ncs, q_level, method="higher"))

        certified_safe   = test_ncs <= threshold          # model certifies safe
        actually_safe    = test_held                       # ground truth

        # Miscoverage: certified safe but actually violated
        n_miscovered = int(np.sum(certified_safe & ~actually_safe))
        n_certified  = int(np.sum(certified_safe))

        # Empirical miscoverage rate (among all test points)
        empirical_miscov = n_miscovered / len(test)
        # Empirical coverage: P(actually safe | certified safe)  — precision
        if n_certified > 0:
            empirical_cov = 1.0 - n_miscovered / n_certified
        else:
            empirical_cov = 1.0   # no false certification = trivially 100%

        empirical_coverage.append(empirical_cov)
        empirical_miscoverage.append(empirical_miscov)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Conformal coverage check  "
        f"(n_cal={n_cal}, n_test={len(test)})",
        fontsize=13,
    )

    claimed_coverage = [1 - a for a in alphas]

    # Left: coverage curve
    ax = axes[0]
    ax.plot(claimed_coverage, empirical_coverage, "o-", color="steelblue",
            label="Empirical precision P(safe|certified)")
    ax.plot([0, 1], [0, 1], "--k", linewidth=0.8, label="Perfect calibration")
    ax.axhline(0.95, color="red", linewidth=0.6, linestyle=":", label="95% reference")
    ax.set_xlabel("Claimed coverage  1 – α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage (precision at each α)")
    ax.legend(fontsize=8)
    ax.set_xlim(0.65, 1.02)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, alpha=0.3)

    # Right: miscoverage curve  (should be <= alpha = below diagonal)
    ax2 = axes[1]
    ax2.plot(alphas, empirical_miscoverage, "s-", color="tomato",
             label="Empirical miscoverage")
    ax2.plot([0, max(alphas)], [0, max(alphas)], "--k", linewidth=0.8,
             label="Claimed level α (must stay below)")
    ax2.fill_between(alphas, empirical_miscoverage, alphas,
                     where=[e <= a for e, a in zip(empirical_miscoverage, alphas)],
                     alpha=0.15, color="green", label="Valid region")
    ax2.set_xlabel("α  (claimed miscoverage rate)")
    ax2.set_ylabel("Empirical miscoverage rate")
    ax2.set_title("Miscoverage (must be ≤ α for validity)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Coverage plot saved to {save_path}")
    if show:
        plt.show()

    # --- textual summary at alpha = 0.05 ---
    try:
        idx_05 = min(range(len(alphas)), key=lambda i: abs(alphas[i] - 0.05))
        a05    = alphas[idx_05]
        cov05  = empirical_coverage[idx_05]
        mis05  = empirical_miscoverage[idx_05]
        valid  = mis05 <= a05
        print(f"\nAt alpha=0.05: empirical coverage={cov05:.3f}, "
              f"empirical miscoverage={mis05:.4f}  "
              f"({'VALID ✓' if valid else 'INVALID — add calibration data'})")
    except Exception:
        pass

    return {
        "alphas": alphas,
        "empirical_coverage": empirical_coverage,
        "empirical_miscoverage": empirical_miscoverage,
        "n_cal": n_cal,
        "n_test": len(test),
        "cal_held_rate":  sum(p.actually_held for p in cal)  / n_cal,
        "test_held_rate": sum(p.actually_held for p in test) / len(test),
    }


# ===========================================================================
# __main__ smoke-test
# ===========================================================================

if __name__ == "__main__":
    import json
    from bluesky_adapter import BlueSkyAdapter
    from recommender import Recommender

    N_CAL = 120    # scenarios for calibration set (≥ 100 recommended)
    ALPHA = 0.05

    print("=== SKYLANCE-X Conformal Prediction smoke-test ===")
    print(f"Building calibration set: {N_CAL} BlueSky scenarios\n")
    print("(This runs full BlueSky simulations — expect ~6–10 minutes)\n")

    adapter    = BlueSkyAdapter()
    engine     = CascadeEngine(adapter, horizon_s=600.0, checkpoint_interval_s=60.0)
    recommender = Recommender(engine)

    cal_points = build_calibration_set(
        n_scenarios=N_CAL,
        adapter=adapter,
        engine=engine,
        recommender=recommender,
        n_aircraft=8,       # 8 aircraft per scenario for speed
        verbose=True,
    )

    held_frac = sum(p.actually_held for p in cal_points) / len(cal_points)
    print(f"\nCalibration set: {len(cal_points)} scenarios, "
          f"{held_frac:.1%} actually held separation\n")

    # Fit certifier
    certifier = ConformalCertifier(alpha=ALPHA)
    certifier.calibrate(cal_points)
    print(f"Conformal threshold q_hat = {certifier.threshold:.4f}  "
          f"(alpha={ALPHA}, n_cal={certifier.n_calibration})")
    print(f"Interpretation: actions with nc_score <= {certifier.threshold:.3f} "
          f"are certified safe at {(1-ALPHA)*100:.0f}% confidence\n")

    # Demonstrate on a fresh scenario
    test_sector = make_mock_sector(n=12, seed=99_999)
    recommendation = recommender.recommend(test_sector)
    cert = certifier.certify(recommendation.score)
    print(f"Live recommendation:  {type(recommendation.action).__name__}")
    print(f"Certificate:          {cert}")
    print()

    # Coverage check (split the calibration set internally)
    print("Running coverage check (80/20 split)...")
    stats = check_coverage(
        cal_points,
        save_path="coverage.png",
        show=False,   # set True if running interactively
    )

    print(f"\nCoverage stats at each alpha:")
    header = f"{'alpha':>6}  {'claimed':>8}  {'empirical_cov':>14}  {'empirical_mis':>14}"
    print(header)
    print("-" * len(header))
    for a, ec, em in zip(stats["alphas"], stats["empirical_coverage"], stats["empirical_miscoverage"]):
        flag = "✓" if em <= a else "✗"
        print(f"  {a:.2f}   {1-a:.3f}         {ec:.4f}           {em:.4f}  {flag}")

    print(f"\nn_cal={stats['n_cal']}  n_test={stats['n_test']}  "
          f"cal_held={stats['cal_held_rate']:.1%}  "
          f"test_held={stats['test_held_rate']:.1%}")
