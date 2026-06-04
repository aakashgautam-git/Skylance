"""
Action recommender for SKYLANCE-X.

Given a SectorState, enumerates up to MAX_CANDIDATES candidate actions using a
greedy priority policy (emergencies first, then swaps, then holds, then vectors),
scores each with the CascadeEngine, and returns the action that maximises a
weighted safety + efficiency + fairness reward against a NoAction baseline.

Reward decomposition
--------------------
  safety     : breach-count reduction × w_safety  +  time-gained × w_time
  fairness   : bonus when the targeted aircraft is emergency / fuel-critical
  efficiency : penalty for holding non-emergency aircraft or large heading deviations

All three terms are in the same reward unit so weights can be tuned uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from state_schema import AircraftState, SectorState
from cascade_engine import (
    CascadeEngine, CascadeScore, CandidateAction,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
)

# ---------------------------------------------------------------------------
# Physical constants (also used in cascade_engine but we don't re-import)
# ---------------------------------------------------------------------------
_NM_TO_M = 1852.0
_EARTH_R_M = 6_371_000.0

MAX_CANDIDATES = 20          # hard cap on candidate set size
_CLOSE_PAIR_THRESHOLD_NM = 20.0  # pairs within this range get vector candidates
_PRIORITY_APPROACH_RADIUS_NM = 80.0  # aircraft within this radius of target airport get held

# Runway → (lat, lon) for nearest-airport selection.  Synced with app.py _AIRPORTS.
_RUNWAY_LOCS: dict[str, tuple[float, float]] = {
    '09R': (13.20, 77.71),  '27L': (13.20, 77.71),   # VOBL Bengaluru
    '09':  (19.09, 72.87),  '27':  (19.09, 72.87),   # VABB Mumbai
    '10':  (28.56, 77.10),  '28':  (28.56, 77.10),   # VIDP Delhi
    '07':  (12.99, 80.17),  '25':  (12.99, 80.17),   # VOMM Chennai
    '09L': (17.24, 78.43),  '27R': (17.24, 78.43),   # VOHS Hyderabad
}


# ===========================================================================
# Reward weights
# ===========================================================================

@dataclass
class RewardWeights:
    """
    Tunable weights for the safety + efficiency + fairness reward.

    Safety dominates: a single breach costs more than any efficiency gain.
    """
    breach_penalty:      float = 500.0   # per breach eliminated vs baseline
    time_gain:           float = 0.5     # per second of additional margin gained
    emergency_bonus:     float = 120.0   # action targets an emergency aircraft
    fuel_critical_bonus: float = 80.0    # action targets a fuel-critical aircraft
    mayday_bonus:        float = 400.0   # AssignRunway to a MAYDAY aircraft (dominates)
    hold_kt_penalty:     float = 0.08    # per kt·minute of speed reduction
    vector_deg_penalty:  float = 0.8     # per degree of heading deviation


# ===========================================================================
# Candidate enumeration
# ===========================================================================

def _urgency(ac: AircraftState) -> tuple:
    """Sort key: smaller → more urgent (for ascending sort)."""
    return (
        not ac.emergency_flag,
        not ac.is_fuel_critical,
        ac.fuel_minutes_above_reserve,
    )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _close_pairs(sector: SectorState, threshold_nm: float) -> list[tuple[AircraftState, AircraftState]]:
    """Return aircraft pairs within threshold_nm, sorted by distance ascending."""
    ac = sector.aircraft
    pairs = []
    for i in range(len(ac)):
        for j in range(i + 1, len(ac)):
            d = _haversine_m(ac[i].lat, ac[i].lon, ac[j].lat, ac[j].lon)
            if d < threshold_nm * _NM_TO_M:
                pairs.append((ac[i], ac[j], d))
    pairs.sort(key=lambda t: t[2])
    return [(t[0], t[1]) for t in pairs]


def enumerate_candidates(
    sector: SectorState,
    max_candidates: int = MAX_CANDIDATES,
) -> list[CandidateAction]:
    """
    Build a priority-ordered candidate list, capped at max_candidates.

    Priority order:
      1. NoAction  (always first — establishes baseline)
      2. Runway assignments for emergency and fuel-critical aircraft
         (all {aircraft} × {available runways} combinations)
      3. Runway swap pairs  — when two emergencies compete for the same
         runway, redirect the lower-urgency one to an uncontested slot
      4. Hold actions for non-critical aircraft (those with most fuel margin)
      5. Vector actions ±30° / ±60° for the lower-priority member of each
         close pair
    """
    candidates: list[CandidateAction] = [NoAction()]
    if len(candidates) >= max_candidates:
        return candidates
    available_runways = [r for r, ok in sector.runway_availability.items() if ok]
    by_urgency = sorted(sector.aircraft, key=_urgency)

    # De-duplication key: (type_name, aircraft_id, detail)
    _seen: set[tuple] = set()

    def _add(c: CandidateAction) -> bool:
        key = (
            type(c).__name__,
            getattr(c, "aircraft_id", ""),
            getattr(c, "runway_id",
                getattr(c, "hold_speed_kt",
                    round(getattr(c, "new_heading_deg", 0), 1))),
        )
        if key in _seen:
            return len(candidates) >= max_candidates
        _seen.add(key)
        candidates.append(c)
        return len(candidates) >= max_candidates

    # --- 0. MAYDAY priority approach — nearest open runway + clear the path --
    # This step runs before general emergency handling so the nearest-runway
    # assignment is always the first non-NoAction candidate.
    mayday_acs = [ac for ac in by_urgency
                  if ac.emergency_flag and ac.emergency_type == "MAYDAY"]
    for ac in mayday_acs:
        if not available_runways:
            continue
        # Nearest open runway by airport great-circle distance
        nearest_rwy = min(
            available_runways,
            key=lambda r: _haversine_m(
                ac.lat, ac.lon, *_RUNWAY_LOCS.get(r, (ac.lat, ac.lon))
            ),
        )
        if _add(AssignRunwayAction(ac.id, nearest_rwy)):
            return candidates
        # Hold non-emergency aircraft within approach radius of target airport
        ap_lat, ap_lon = _RUNWAY_LOCS.get(nearest_rwy, (ac.lat, ac.lon))
        for other in sector.aircraft:
            if other.id == ac.id or other.emergency_flag or other.is_fuel_critical:
                continue
            if (_haversine_m(other.lat, other.lon, ap_lat, ap_lon) / _NM_TO_M
                    < _PRIORITY_APPROACH_RADIUS_NM):
                if _add(HoldAction(other.id)):
                    return candidates

    # --- 1. Greedy runway assignments for priority aircraft ------------------
    for ac in by_urgency:
        if ac.emergency_flag or ac.is_fuel_critical:
            for rwy in available_runways:
                if _add(AssignRunwayAction(ac.id, rwy)):
                    return candidates

    # --- 2. Runway swaps: when two emergency aircraft compete for the same
    #        runway, redirect the lower-urgency one to an uncontested slot.
    # `occupied` = runways already declared as runway_needed by any needy
    # aircraft.  We never redirect onto an occupied runway — that shifts the
    # conflict rather than resolving it.
    needy = [ac for ac in by_urgency if ac.runway_needed and ac.emergency_flag]
    occupied = {ac.runway_needed for ac in needy}
    for i in range(len(needy)):
        for j in range(i + 1, len(needy)):
            a, b = needy[i], needy[j]   # a has higher urgency (earlier in by_urgency)
            if a.runway_needed != b.runway_needed:
                continue  # different runways — no contention to resolve
            # Both a and b are competing for the same runway.
            # a has higher urgency so a keeps it; redirect b to any available
            # runway that no other needy aircraft is already competing for.
            for rwy in available_runways:
                if rwy not in occupied:
                    if _add(AssignRunwayAction(b.id, rwy)):
                        return candidates

    # --- 3. Hold actions for non-critical aircraft (most fuel = safest to hold)
    holdable = sorted(
        [ac for ac in sector.aircraft if not ac.emergency_flag and not ac.is_fuel_critical],
        key=lambda a: -a.fuel_minutes_above_reserve,
    )
    for ac in holdable[:5]:
        if _add(HoldAction(ac.id)):
            return candidates

    # --- 4. Vector actions for close pairs ----------------------------------
    for ac1, ac2 in _close_pairs(sector, _CLOSE_PAIR_THRESHOLD_NM)[:4]:
        # Vector the lower-urgency aircraft of the pair
        lower = ac2 if _urgency(ac1) <= _urgency(ac2) else ac1
        for delta in [30, -30, 60, -60]:
            new_hdg = (lower.heading_deg + delta) % 360.0
            if _add(VectorAction(lower.id, new_hdg)):
                return candidates

    return candidates[:max_candidates]


# ===========================================================================
# Reward
# ===========================================================================

def _heading_delta(hdg_a: float, hdg_b: float) -> float:
    """Smallest signed angle between two headings, in [0, 180]."""
    return abs(((hdg_a - hdg_b) + 180) % 360 - 180)


def compute_reward(
    score: CascadeScore,
    action: CandidateAction,
    sector: SectorState,
    baseline: CascadeScore,
    weights: RewardWeights,
) -> float:
    """
    Scalar reward for (action, score) relative to baseline NoAction.

    Higher = better.  Safety is the dominant term; efficiency and fairness
    break ties.
    """
    horizon = score.horizon_s

    # --- Safety ---
    breach_improvement = baseline.total_breach_count - score.total_breach_count
    time_baseline = baseline.time_to_first_breach_s if baseline.time_to_first_breach_s is not None else horizon
    time_score    = score.time_to_first_breach_s    if score.time_to_first_breach_s    is not None else horizon
    safety = (
        breach_improvement * weights.breach_penalty
        + (time_score - time_baseline) * weights.time_gain
    )

    # --- Fairness ---
    fairness = 0.0
    target_id: Optional[str] = getattr(action, "aircraft_id", None)
    if target_id:
        target = next((ac for ac in sector.aircraft if ac.id == target_id), None)
        if target:
            if target.emergency_flag:
                fairness += weights.emergency_bonus
            if target.is_fuel_critical:
                fairness += weights.fuel_critical_bonus
            # MAYDAY priority-approach bonus dominates all other reward terms
            if (target.emergency_type == "MAYDAY"
                    and isinstance(action, AssignRunwayAction)):
                fairness += weights.mayday_bonus

    # --- Efficiency ---
    efficiency = 0.0
    if isinstance(action, HoldAction):
        target = next((ac for ac in sector.aircraft if ac.id == action.aircraft_id), None)
        if target:
            kt_reduction = max(0.0, target.ground_speed_kt - action.hold_speed_kt)
            # Cost: kt_reduction × hold_duration_min (assume holds until horizon)
            hold_min = horizon / 60.0
            efficiency -= kt_reduction * hold_min * weights.hold_kt_penalty
    elif isinstance(action, VectorAction):
        target = next((ac for ac in sector.aircraft if ac.id == action.aircraft_id), None)
        if target:
            delta = _heading_delta(action.new_heading_deg, target.heading_deg)
            efficiency -= delta * weights.vector_deg_penalty

    return safety + fairness + efficiency


# ===========================================================================
# Recommendation result
# ===========================================================================

@dataclass
class Recommendation:
    action: CandidateAction
    score: CascadeScore
    reward: float
    baseline_score: CascadeScore
    candidates_evaluated: int

    @property
    def action_type(self) -> str:
        return type(self.action).__name__

    @property
    def improvement_summary(self) -> str:
        b_total = self.baseline_score.total_breach_count
        r_total = self.score.total_breach_count
        if b_total == 0 and r_total == 0:
            return "No improvement needed — sector is already safe"
        if r_total < b_total:
            return (
                f"Reduces breaches {b_total} → {r_total}; "
                f"first breach at t+{self.score.time_to_first_breach_s or 'none':.0f}s"
                if self.score.time_to_first_breach_s else
                f"Reduces breaches {b_total} → {r_total}; no breach within horizon"
            )
        return f"No breach reduction (baseline {b_total}, this action {r_total})"

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "action": {k: v for k, v in vars(self.action).items()} if hasattr(self.action, '__dict__') else {},
            "reward": round(self.reward, 2),
            "candidates_evaluated": self.candidates_evaluated,
            "score": self.score.to_dict(),
            "baseline": self.baseline_score.to_dict(),
            "improvement_summary": self.improvement_summary,
        }


# ===========================================================================
# Recommender
# ===========================================================================

class Recommender:
    """
    Main entry point for the SKYLANCE-X action suggestion layer.

    Usage:
        rec = Recommender(engine)
        result = rec.recommend(sector)
        print(result.action, result.score)
    """

    def __init__(
        self,
        engine: CascadeEngine,
        weights: Optional[RewardWeights] = None,
        max_candidates: int = MAX_CANDIDATES,
    ) -> None:
        self.engine = engine
        self.weights = weights or RewardWeights()
        self.max_candidates = max_candidates

    def recommend(self, sector: SectorState) -> Recommendation:
        """
        Enumerate candidates, score each via CascadeEngine, pick highest reward.

        Returns a Recommendation containing the chosen action, its CascadeScore,
        the computed reward, and the NoAction baseline for comparison.
        """
        candidates = enumerate_candidates(sector, self.max_candidates)

        # Score baseline once; reuse for all reward computations
        baseline = self.engine.evaluate(sector, NoAction())

        best_action: CandidateAction = NoAction()
        best_score:  CascadeScore    = baseline
        best_reward: float           = -math.inf

        for action in candidates:
            if isinstance(action, NoAction):
                score = baseline
            else:
                score = self.engine.evaluate(sector, action)

            rew = compute_reward(score, action, sector, baseline, self.weights)
            if rew > best_reward:
                best_reward = rew
                best_action = action
                best_score  = score

        return Recommendation(
            action=best_action,
            score=best_score,
            reward=best_reward,
            baseline_score=baseline,
            candidates_evaluated=len(candidates),
        )

    def recommend_top_k(self, sector: SectorState, k: int = 3) -> list[Recommendation]:
        """Return the top-k actions by reward (useful for presenting alternatives)."""
        candidates = enumerate_candidates(sector, self.max_candidates)
        baseline   = self.engine.evaluate(sector, NoAction())

        scored: list[tuple[float, CandidateAction, CascadeScore]] = []
        for action in candidates:
            score = baseline if isinstance(action, NoAction) else self.engine.evaluate(sector, action)
            rew   = compute_reward(score, action, sector, baseline, self.weights)
            scored.append((rew, action, score))

        scored.sort(key=lambda t: -t[0])
        return [
            Recommendation(action=a, score=s, reward=r,
                           baseline_score=baseline, candidates_evaluated=len(candidates))
            for r, a, s in scored[:k]
        ]


# ===========================================================================
# __main__ smoke-test
# ===========================================================================

if __name__ == "__main__":
    import json
    from state_schema import make_mock_sector
    from bluesky_adapter import BlueSkyAdapter

    print("=== SKYLANCE-X Recommender smoke-test ===\n")

    sector  = make_mock_sector(n=12, seed=42)
    adapter = BlueSkyAdapter()
    engine  = CascadeEngine(adapter, horizon_s=600.0, checkpoint_interval_s=60.0)
    rec     = Recommender(engine)

    print("Sector (sorted by urgency):")
    for ac in sorted(sector.aircraft, key=_urgency):
        mins = f"{ac.fuel_minutes_above_reserve:.0f}min" if ac.fuel_minutes_above_reserve < 999 else "∞"
        flags = []
        if ac.emergency_flag:   flags.append(ac.emergency_type or "EMG")
        if ac.is_fuel_critical: flags.append("CRITICAL")
        print(f"  {ac.id:<8}  FL{ac.altitude_ft/100:03.0f}  {ac.ground_speed_kt:.0f}kt  "
              f"fuel_margin={mins}  {' '.join(flags)}")

    print(f"\nEnumerating candidates (max {MAX_CANDIDATES})...")
    candidates = enumerate_candidates(sector)

    # --- Candidate correctness assertions ------------------------------------
    _open_rwys = {r for r, ok in sector.runway_availability.items() if ok}
    _seen_keys: set[tuple] = set()
    for _c in candidates:
        if isinstance(_c, NoAction):
            _key: tuple = ("NoAction",)
        elif isinstance(_c, AssignRunwayAction):
            assert _c.runway_id in sector.runway_availability, (
                f"Unknown runway {_c.runway_id!r} in candidate")
            assert sector.runway_availability[_c.runway_id], (
                f"Candidate assigns {_c.aircraft_id} to closed runway {_c.runway_id!r}")
            _key = ("AssignRunwayAction", _c.aircraft_id, _c.runway_id)
        elif isinstance(_c, HoldAction):
            _key = ("HoldAction", _c.aircraft_id)
        elif isinstance(_c, VectorAction):
            _key = ("VectorAction", _c.aircraft_id, round(_c.new_heading_deg, 1))
        else:
            _key = (type(_c).__name__,)
        assert _key not in _seen_keys, f"Duplicate candidate: {_key}"
        _seen_keys.add(_key)
    print(f"  Assertions passed: {len(candidates)} candidates, "
          f"0 closed-runway assignments, 0 duplicates.")

    for i, c in enumerate(candidates):
        print(f"  [{i:2d}] {type(c).__name__:<22} "
              f"{getattr(c, 'aircraft_id', ''):<8} "
              f"{getattr(c, 'runway_id', getattr(c, 'new_heading_deg', getattr(c, 'hold_speed_kt', '')))}")

    print(f"\nRunning recommender ({len(candidates)} candidates × 10 checkpoints)...")
    result = rec.recommend(sector)

    print(f"\n=== Recommendation ===")
    print(json.dumps(result.to_dict(), indent=2))

    print(f"\n=== Top-3 alternatives ===")
    top3 = rec.recommend_top_k(sector, k=3)
    for i, r in enumerate(top3, 1):
        print(f"  #{i}  reward={r.reward:+.1f}  {r.action_type}  {r.improvement_summary}")

    # --- Swap-logic targeted test -------------------------------------------
    # Two emergencies both want 27L; only 27R and 09L are uncontested open slots.
    # Swap step must redirect EMG002 (lower urgency) to 27R or 09L only —
    # never to 27L (contested) or 09R (closed).
    print("\n=== Swap-logic targeted test ===")
    from state_schema import AircraftState, SectorState as _SS

    _swap_sector = _SS(
        aircraft=[
            AircraftState("EMG001", 52.0, 4.0, 30000, 90,  450,
                          2500.0, 10.0, 2200.0, True,  "MAYDAY",   "27L"),
            AircraftState("EMG002", 52.5, 4.5, 32000, 270, 420,
                          2800.0,  9.0, 2100.0, True,  "PAN-PAN",  "27L"),
            AircraftState("NRM001", 51.8, 4.8, 34000, 180, 430,
                          6000.0, 11.0, 2200.0, False,  None,       None),
        ],
        runway_availability={"27L": True, "27R": True, "09L": True, "09R": False},
        sim_time_s=0.0,
    )

    _swap_cands = enumerate_candidates(_swap_sector)
    _swap_open  = {r for r, ok in _swap_sector.runway_availability.items() if ok}
    _occupied   = {"27L"}   # the only runway_needed held by needy aircraft

    # 1. No closed-runway assignment
    for _c in _swap_cands:
        if isinstance(_c, AssignRunwayAction):
            assert _c.runway_id in _swap_sector.runway_availability, \
                f"Unknown runway {_c.runway_id!r}"
            assert _swap_sector.runway_availability[_c.runway_id], \
                f"Closed runway {_c.runway_id!r} in candidate for {_c.aircraft_id}"

    # 2. No duplicate candidates
    _swap_keys: set[tuple] = set()
    for _c in _swap_cands:
        if isinstance(_c, NoAction):
            _k: tuple = ("NoAction",)
        elif isinstance(_c, AssignRunwayAction):
            _k = ("AssignRunwayAction", _c.aircraft_id, _c.runway_id)
        elif isinstance(_c, HoldAction):
            _k = ("HoldAction", _c.aircraft_id)
        elif isinstance(_c, VectorAction):
            _k = ("VectorAction", _c.aircraft_id, round(_c.new_heading_deg, 1))
        else:
            _k = (type(_c).__name__,)
        assert _k not in _swap_keys, f"Duplicate candidate in swap test: {_k}"
        _swap_keys.add(_k)

    # 3. Swap step must not redirect EMG002 to an occupied runway via its own
    #    path (the contested "27L" is in _occupied so it is excluded by the
    #    `rwy not in occupied` guard; any such candidate came from step 1 only).
    #    Both EMG001 and EMG002 should have 27R and 09L as options (from step 1).
    _emg002_rwys = {_c.runway_id for _c in _swap_cands
                    if isinstance(_c, AssignRunwayAction) and _c.aircraft_id == "EMG002"}
    assert "09R" not in _emg002_rwys, "Closed runway 09R must never appear"
    assert _emg002_rwys == {"27L", "27R", "09L"}, (
        f"EMG002 should have options {{27L, 27R, 09L}}, got {_emg002_rwys}")

    print(f"  EMG002 runway options: {sorted(_emg002_rwys)}")
    print(f"  {len(_swap_cands)} candidates, 0 duplicates, 0 closed-runway assignments.")
    print("  All swap assertions passed.")
