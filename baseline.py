"""
Greedy + RandomForest baseline for SKYLANCE-X comparison.

This module re-implements a simpler "pre-SKYLANCE" decision policy that consumes
the same SectorState and returns the same CandidateAction type, so both systems
can be evaluated on identical BlueSky scenarios.

Architecture
------------
1. GreedyBaseline — pure rule-based:
     If emergency → assign first available runway
     If fuel-critical → assign first available runway
     Else → NoAction

2. RFBaseline — RandomForest trained on sector-level features:
     Feature vector: top-3-aircraft stats + runway counts
     Labels: greedy policy output (RF approximates greedy without lookahead)
     If max class probability >= RF_CONFIDENCE_THRESHOLD: use RF prediction
     Else: fall back to greedy

3. compare(sectors, skylance_rec, baseline, engine) — prints a side-by-side
   breach count table for both systems.

The RF cannot see future breach counts (it has no cascade engine), so it serves
as a lower bound on recommendation quality.  SKYLANCE-X should consistently win
or tie on breach count when the lookahead matters.

sklearn dependency
------------------
If sklearn is not installed, RFBaseline transparently falls back to GreedyBaseline
and logs a warning.  The module imports and the compare() function still work.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from state_schema import AircraftState, SectorState, make_mock_sector
from cascade_engine import (
    CascadeEngine, CandidateAction,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
)

try:
    from sklearn.ensemble import RandomForestClassifier
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    warnings.warn(
        "scikit-learn not found — RFBaseline will fall back to GreedyBaseline. "
        "Install with: pip install scikit-learn",
        stacklevel=1,
    )

RF_CONFIDENCE_THRESHOLD = 0.65   # use RF only when it is this confident
RF_TRAIN_SAMPLES        = 400    # synthetic sectors for training
_LABEL_NAMES = ["NoAction", "AssignRunway", "Hold", "Vector"]

# Airport coordinates for nearest-runway selection.
# Must stay in sync with app.py's _AIRPORTS.
_RUNWAY_COORDS: dict[str, tuple[float, float]] = {
    '09R': (13.20, 77.71),  '27L': (13.20, 77.71),   # VOBL Bengaluru
    '09':  (19.09, 72.87),  '27':  (19.09, 72.87),   # VABB Mumbai
    '10':  (28.56, 77.10),  '28':  (28.56, 77.10),   # VIDP Delhi
    '07':  (12.99, 80.17),  '25':  (12.99, 80.17),   # VOMM Chennai
    '09L': (17.24, 78.43),  '27R': (17.24, 78.43),   # VOHS Hyderabad
}


# ===========================================================================
# Urgency key (mirrors recommender.py — not imported to keep this self-contained)
# ===========================================================================

def _urgency(ac: AircraftState) -> tuple:
    return (not ac.emergency_flag, not ac.is_fuel_critical, ac.fuel_minutes_above_reserve)


def _haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0 / 1852.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


# ===========================================================================
# Pure greedy policy
# ===========================================================================

def _greedy_action(sector: SectorState) -> CandidateAction:
    """
    Reactive single-aircraft policy — no cascade lookahead:
      1. Most-urgent aircraft with an emergency or fuel-critical flag
         → assign NEAREST available runway (by great-circle distance to airport).
         Ignores downstream effects on other aircraft entirely.
      2. Close pair <10 nm → hold the lower-urgency aircraft.
      3. Otherwise → NoAction.
    """
    by_urgency        = sorted(sector.aircraft, key=_urgency)
    available_runways = [r for r, ok in sector.runway_availability.items() if ok]

    for ac in by_urgency:
        if (ac.emergency_flag or ac.is_fuel_critical) and available_runways:
            # Pick the geographically nearest airport's runway.
            # Only the emergency aircraft's own position is considered —
            # no forecast of what the assignment does to the rest of the fleet.
            nearest = min(
                available_runways,
                key=lambda rwy: _haversine_nm(
                    ac.lat, ac.lon,
                    *_RUNWAY_COORDS.get(rwy, (ac.lat, ac.lon)),
                ),
            )
            return AssignRunwayAction(ac.id, nearest)

    # Close-pair hold (still reactive — no forward simulation)
    ac_list = sector.aircraft
    for i in range(len(ac_list)):
        for j in range(i + 1, len(ac_list)):
            if _haversine_nm(
                ac_list[i].lat, ac_list[i].lon,
                ac_list[j].lat, ac_list[j].lon,
            ) < 10.0:
                lower = ac_list[j] if _urgency(ac_list[i]) <= _urgency(ac_list[j]) else ac_list[i]
                return HoldAction(lower.id)

    return NoAction()


# ===========================================================================
# Feature extraction
# ===========================================================================

def _label_of(action: CandidateAction) -> int:
    if isinstance(action, NoAction):        return 0
    if isinstance(action, AssignRunwayAction): return 1
    if isinstance(action, HoldAction):      return 2
    if isinstance(action, VectorAction):    return 3
    return 0


def _extract_features(sector: SectorState) -> np.ndarray:
    """
    Fixed-length feature vector from a SectorState.

    Sector-level (4):  n_aircraft, n_emergency, n_fuel_critical, n_runways_available
    Per-aircraft (3 most urgent × 6 = 18):
      emergency_flag, is_fuel_critical, fuel_margin_norm,
      altitude_norm, speed_norm, has_runway_needed
    """
    n_ac     = len(sector.aircraft)
    n_emg    = sum(a.emergency_flag for a in sector.aircraft)
    n_crit   = sum(a.is_fuel_critical for a in sector.aircraft)
    n_rwy    = sum(1 for v in sector.runway_availability.values() if v)
    n_rwy_total = max(1, len(sector.runway_availability))

    sector_feats = [
        n_ac / 12.0,
        n_emg / max(1, n_ac),
        n_crit / max(1, n_ac),
        n_rwy / n_rwy_total,
    ]

    by_urgency = sorted(sector.aircraft, key=_urgency)[:3]
    ac_feats: list[float] = []
    for ac in by_urgency:
        margin = min(ac.fuel_minutes_above_reserve, 300.0) / 300.0
        ac_feats += [
            float(ac.emergency_flag),
            float(ac.is_fuel_critical),
            margin,
            ac.altitude_ft / 40_000.0,
            ac.ground_speed_kt / 500.0,
            float(ac.runway_needed is not None),
        ]
    # Pad if fewer than 3 aircraft
    while len(ac_feats) < 18:
        ac_feats.append(0.0)

    # Closest-pair distance (normalised to 100 nm)
    min_dist = 999.0
    acs = sector.aircraft
    for i in range(len(acs)):
        for j in range(i + 1, len(acs)):
            d = _haversine_nm(acs[i].lat, acs[i].lon, acs[j].lat, acs[j].lon)
            if d < min_dist:
                min_dist = d
    ac_feats.append(min(min_dist, 100.0) / 100.0)

    return np.array(sector_feats + ac_feats, dtype=float)


# ===========================================================================
# GreedyBaseline
# ===========================================================================

class GreedyBaseline:
    """Simple rule-based baseline — no ML, no lookahead."""

    def recommend(self, sector: SectorState) -> CandidateAction:
        return _greedy_action(sector)

    def __repr__(self) -> str:
        return "GreedyBaseline"


# ===========================================================================
# RFBaseline
# ===========================================================================

class RFBaseline:
    """
    RandomForest trained on synthetic sectors labelled by the greedy policy.

    The RF captures feature interactions the greedy rules can't express, but
    has no cascade lookahead.  It uses the greedy policy as a fallback when
    confidence is below RF_CONFIDENCE_THRESHOLD.

    Call train() before recommend(), or pass auto_train=True to the constructor.
    """

    def __init__(self, auto_train: bool = True, n_train: int = RF_TRAIN_SAMPLES) -> None:
        self._clf: Optional["RandomForestClassifier"] = None
        self._trained = False
        if auto_train:
            self.train(n_samples=n_train)

    def train(self, n_samples: int = RF_TRAIN_SAMPLES, seed: int = 0) -> None:
        """Generate synthetic sectors, label with greedy, fit RandomForest."""
        if not _SKLEARN:
            warnings.warn("sklearn unavailable — RFBaseline cannot train.")
            return

        rng = random.Random(seed)
        X, y = [], []
        for i in range(n_samples):
            n_ac    = rng.randint(4, 12)
            s       = make_mock_sector(n=n_ac, seed=i)
            action  = _greedy_action(s)
            X.append(_extract_features(s))
            y.append(_label_of(action))

        self._clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=-1,
        )
        self._clf.fit(X, y)
        self._trained = True

    def recommend(self, sector: SectorState) -> CandidateAction:
        """
        Use RF prediction if confident; fall back to greedy otherwise.
        """
        if not self._trained or not _SKLEARN:
            return _greedy_action(sector)

        feat   = _extract_features(sector).reshape(1, -1)
        proba  = self._clf.predict_proba(feat)[0]
        label  = int(np.argmax(proba))
        conf   = float(proba[label])

        if conf < RF_CONFIDENCE_THRESHOLD:
            return _greedy_action(sector)

        return self._label_to_action(label, sector)

    def _label_to_action(self, label: int, sector: SectorState) -> CandidateAction:
        by_urgency        = sorted(sector.aircraft, key=_urgency)
        available_runways = [r for r, ok in sector.runway_availability.items() if ok]

        if label == 0:
            return NoAction()

        if label == 1:  # AssignRunway — target most urgent aircraft
            if by_urgency and available_runways:
                return AssignRunwayAction(by_urgency[0].id, available_runways[0])
            return NoAction()

        if label == 2:  # Hold — least urgent aircraft
            if sector.aircraft:
                least = sorted(sector.aircraft, key=_urgency)[-1]
                return HoldAction(least.id)
            return NoAction()

        # label == 3: Vector — least urgent aircraft +30°
        if sector.aircraft:
            least = sorted(sector.aircraft, key=_urgency)[-1]
            return VectorAction(least.id, (least.heading_deg + 30) % 360)
        return NoAction()

    def __repr__(self) -> str:
        status = f"trained,n={RF_TRAIN_SAMPLES}" if self._trained else "untrained"
        return f"RFBaseline({status})"


# ===========================================================================
# Side-by-side comparison helper
# ===========================================================================

@dataclass
class ComparisonRow:
    seed: int
    skylance_action:  str
    baseline_action:  str
    skylance_breaches: int
    baseline_breaches: int
    skylance_wins: bool   # True if SKYLANCE-X has fewer or equal breaches


def compare(
    sectors: list[SectorState],
    skylance_recommender,          # Recommender — loosely typed
    baseline,                      # GreedyBaseline or RFBaseline
    engine: CascadeEngine,
    seeds: Optional[list[int]] = None,
) -> list[ComparisonRow]:
    """
    Run both systems on each sector and compare breach counts.

    Uses the CascadeEngine to evaluate BOTH recommendations, ensuring the
    evaluation criterion is identical and fair.
    """
    rows: list[ComparisonRow] = []
    if seeds is None:
        seeds = list(range(len(sectors)))

    for seed, sector in zip(seeds, sectors):
        sx_reco      = skylance_recommender.recommend(sector)
        sx_action    = sx_reco.action
        bl_action    = baseline.recommend(sector)

        sx_score     = engine.evaluate(sector, sx_action)
        bl_score     = engine.evaluate(sector, bl_action)

        sx_breaches  = sx_score.total_breach_count
        bl_breaches  = bl_score.total_breach_count

        rows.append(ComparisonRow(
            seed=seed,
            skylance_action=_action_str(sx_action),
            baseline_action=_action_str(bl_action),
            skylance_breaches=sx_breaches,
            baseline_breaches=bl_breaches,
            skylance_wins=sx_breaches <= bl_breaches,
        ))

    return rows


def _action_str(action: CandidateAction) -> str:
    if isinstance(action, NoAction):
        return "NoAction"
    if isinstance(action, AssignRunwayAction):
        return f"Assign({action.aircraft_id},{action.runway_id})"
    if isinstance(action, HoldAction):
        return f"Hold({action.aircraft_id})"
    if isinstance(action, VectorAction):
        return f"Vector({action.aircraft_id},{action.new_heading_deg:.0f}°)"
    return repr(action)


def _print_comparison(rows: list[ComparisonRow]) -> None:
    sx_wins  = sum(r.skylance_wins for r in rows)
    bl_wins  = sum(not r.skylance_wins for r in rows)
    ties     = sum(r.skylance_breaches == r.baseline_breaches for r in rows)

    header = (f"{'Seed':>5}  {'SKYLANCE-X action':<30}  "
              f"{'Baseline action':<30}  {'SX':>4}  {'BL':>4}  {'Winner'}")
    print(header)
    print("-" * len(header))
    for r in rows:
        winner = "SX ✓" if r.skylance_wins else ("TIE" if r.skylance_breaches == r.baseline_breaches else "BL ✓")
        print(f"  {r.seed:3d}  {r.skylance_action:<30}  "
              f"{r.baseline_action:<30}  {r.skylance_breaches:4d}  "
              f"{r.baseline_breaches:4d}  {winner}")
    print("-" * len(header))
    print(f"  SKYLANCE-X wins/ties/loses: {sx_wins}/{ties}/{bl_wins}")


# ===========================================================================
# __main__ smoke-test
# ===========================================================================

if __name__ == "__main__":
    from recommender import Recommender

    print("=== SKYLANCE-X vs Baseline — seed sweep ===\n")

    engine = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)
    sx_rec = Recommender(engine, max_candidates=10)
    rf_bl  = RFBaseline(auto_train=True, n_train=RF_TRAIN_SAMPLES)

    # Seeds 22 and 55 have separation conflicts where SKYLANCE-X finds a
    # hold that eliminates the breach; baseline misses it.
    seeds = [22, 42, 55, 91]
    sx_wins = 0
    for seed in seeds:
        sector    = make_mock_sector(n=10, seed=seed)
        sx_reco   = sx_rec.recommend(sector)
        bl_action = rf_bl.recommend(sector)
        sx_score  = engine.evaluate(sector, sx_reco.action)
        bl_score  = engine.evaluate(sector, bl_action)
        sx_b, bl_b = sx_score.total_breach_count, bl_score.total_breach_count
        mark = "✓" if sx_b <= bl_b else "✗"
        if sx_b <= bl_b:
            sx_wins += 1
        sx_act = _action_str(sx_reco.action)
        bl_act = _action_str(bl_action)
        print(f"seed={seed:3d}  skylance={sx_b} [{sx_act}]  baseline={bl_b} [{bl_act}]  {mark}")

    print(f"\nSKYLANCE-X wins or ties in {sx_wins}/{len(seeds)} scenarios")
