"""
Counterfactual minimal-perturbation explainer for SKYLANCE-X.

Given the chosen action and the current SectorState, finds the single-feature
change of smallest normalized magnitude that would flip the recommendation to a
different action, and returns it as a plain-English sentence.

Search strategy
---------------
Perturbations are tried cheapest-first by normalized cost:

  1. Analytical fuel/reserve crossings (cost = Δkg / reserve_kg; often < 0.1)
  2. Binary toggles: emergency_flag, runway_availability (cost = 1.0 each)
  3. Emergency-type promotions/demotions (cost = 1.0)

For the analytically computable crossings (fuel crossing the is_fuel_critical
30-minute threshold), we avoid expensive binary search and instead compute the
exact minimum delta, then verify with one recommender call.

Normalized cost formula
-----------------------
  fuel_kg change:          |Δfuel| / reserve_fuel_kg
  reserve_fuel_kg change:  |Δreserve| / reserve_fuel_kg
  burn_rate change:        |Δrate| / original_rate
  binary toggle:           1.0

The counterfactual with the lowest normalized cost is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from state_schema import AircraftState, SectorState
from cascade_engine import (
    CandidateAction,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
)
from recommender import Recommender, Recommendation

# ε added to threshold crossings so the perturbed value is strictly inside
# the new region, not sitting on the boundary.
_THRESHOLD_EPS = 0.5    # kg
_HOLD_SPEED_MIN = 180.0 # kt — floor for realistic holding speeds


# ===========================================================================
# Action identity string  (used to detect a flip)
# ===========================================================================

def _action_key(action: CandidateAction) -> str:
    """Canonical string describing the action; two equal actions produce the same key."""
    if isinstance(action, NoAction):
        return "NoAction"
    if isinstance(action, AssignRunwayAction):
        return f"AssignRunway({action.aircraft_id},{action.runway_id})"
    if isinstance(action, HoldAction):
        return f"Hold({action.aircraft_id})"
    if isinstance(action, VectorAction):
        return f"Vector({action.aircraft_id},{action.new_heading_deg:.0f})"
    return repr(action)


def _action_label(action: CandidateAction) -> str:
    """Verb phrase for the action, suitable for 'the system would <label>'."""
    if isinstance(action, NoAction):
        return "take no action"
    if isinstance(action, AssignRunwayAction):
        return f"assign {action.aircraft_id} to runway {action.runway_id}"
    if isinstance(action, HoldAction):
        return f"place {action.aircraft_id} in a hold"
    if isinstance(action, VectorAction):
        return f"vector {action.aircraft_id} to {action.new_heading_deg:.0f}°"
    return repr(action)


# ===========================================================================
# Sector perturbation helpers  (pure — never mutate input)
# ===========================================================================

def _copy_aircraft(ac: AircraftState, **overrides) -> AircraftState:
    d = {
        "id": ac.id, "lat": ac.lat, "lon": ac.lon,
        "altitude_ft": ac.altitude_ft, "heading_deg": ac.heading_deg,
        "ground_speed_kt": ac.ground_speed_kt, "fuel_kg": ac.fuel_kg,
        "fuel_burn_rate_kg_per_min": ac.fuel_burn_rate_kg_per_min,
        "reserve_fuel_kg": ac.reserve_fuel_kg,
        "emergency_flag": ac.emergency_flag, "emergency_type": ac.emergency_type,
        "runway_needed": ac.runway_needed,
    }
    d.update(overrides)
    return AircraftState(**d)


def _patch_aircraft(sector: SectorState, aircraft_id: str, **overrides) -> SectorState:
    """Return a new SectorState with one aircraft's fields changed."""
    return SectorState(
        aircraft=[
            _copy_aircraft(ac, **overrides) if ac.id == aircraft_id else ac
            for ac in sector.aircraft
        ],
        runway_availability=dict(sector.runway_availability),
        sim_time_s=sector.sim_time_s,
    )


def _patch_runway(sector: SectorState, runway_id: str, available: bool) -> SectorState:
    """Return a new SectorState with one runway's availability changed."""
    rwa = dict(sector.runway_availability)
    rwa[runway_id] = available
    return SectorState(
        aircraft=list(sector.aircraft),
        runway_availability=rwa,
        sim_time_s=sector.sim_time_s,
    )


# ===========================================================================
# Counterfactual record
# ===========================================================================

@dataclass
class Counterfactual:
    feature: str               # e.g. "KLM792.fuel_kg"
    original_value: Any        # value before perturbation
    perturbed_value: Any       # value that causes the flip
    normalized_cost: float     # |Δ| / scale; lower = more minimal
    original_action: str       # key of original recommendation
    counterfactual_action: str # key of new recommendation after perturbation
    sentence: str              # plain-English explanation


# ===========================================================================
# Sentence generator
# ===========================================================================

def _sentence(
    feature: str,
    original_value: Any,
    perturbed_value: Any,
    original_action: CandidateAction,
    new_action: CandidateAction,
    ac: Optional[AircraftState] = None,
) -> str:
    orig_label = _action_label(original_action)
    new_label  = _action_label(new_action)

    # --- fuel_kg ---
    if feature.endswith(".fuel_kg") and ac is not None:
        delta = perturbed_value - original_value
        sign  = "increased" if delta > 0 else "decreased"
        from_margin = ac.fuel_minutes_above_reserve
        burn = ac.fuel_burn_rate_kg_per_min
        to_margin = (perturbed_value - ac.reserve_fuel_kg) / burn if burn > 0 else 0
        return (
            f"If {ac.id}'s fuel {sign} by {abs(delta):.0f} kg "
            f"(from {original_value:.0f} kg to {perturbed_value:.0f} kg, "
            f"raising margin from {from_margin:.1f} to {to_margin:.1f} min above reserve), "
            f"the system would {new_label} instead of {orig_label}."
        )

    # --- reserve_fuel_kg ---
    if feature.endswith(".reserve_fuel_kg") and ac is not None:
        delta = perturbed_value - original_value
        sign  = "increased" if delta > 0 else "decreased"
        return (
            f"If {ac.id}'s regulatory reserve changed by {delta:+.0f} kg "
            f"(from {original_value:.0f} kg to {perturbed_value:.0f} kg), "
            f"the system would {new_label} instead of {orig_label}."
        )

    # --- emergency_flag ---
    if feature.endswith(".emergency_flag") and ac is not None:
        if perturbed_value:
            return (
                f"If {ac.id} declared an emergency, "
                f"the system would {new_label} instead of {orig_label}."
            )
        else:
            return (
                f"If {ac.id}'s emergency were resolved, "
                f"the system would {new_label} instead of {orig_label}."
            )

    # --- emergency_type ---
    if feature.endswith(".emergency_type") and ac is not None:
        if perturbed_value is None:
            return (
                f"If {ac.id}'s emergency type were cleared, "
                f"the system would {new_label} instead of {orig_label}."
            )
        return (
            f"If {ac.id}'s emergency were upgraded to {perturbed_value}, "
            f"the system would {new_label} instead of {orig_label}."
        )

    # --- runway_availability ---
    if "runway_availability" in feature:
        rwy   = feature.split(".")[-1]
        state = "closed" if not perturbed_value else "opened"
        return (
            f"If runway {rwy} were {state}, "
            f"the system would {new_label} instead of {orig_label}."
        )

    # --- generic fallback ---
    return (
        f"If {feature} changed from {original_value!r} to {perturbed_value!r}, "
        f"the system would {new_label} instead of {orig_label}."
    )


# ===========================================================================
# Core explainer
# ===========================================================================

class Explainer:
    """
    Finds the minimal single-feature perturbation that flips the recommendation.

    Usage:
        explainer = Explainer(recommender)
        cf = explainer.explain(sector, recommendation)
        if cf:
            print(cf.sentence)
    """

    def __init__(self, recommender: Recommender) -> None:
        self.recommender = recommender

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self, sector: SectorState, recommendation: Recommendation
    ) -> Optional[Counterfactual]:
        """
        Return the minimal-cost Counterfactual, or None if no single-feature
        flip was found within realistic bounds.
        """
        original_action = recommendation.action
        original_key    = _action_key(original_action)

        candidates: list[Counterfactual] = []

        # --- Phase 1: analytical fuel / reserve crossings (cheapest) --------
        for ac in sector.aircraft:
            candidates.extend(
                self._fuel_crossings(sector, ac, original_action, original_key)
            )
            candidates.extend(
                self._reserve_crossings(sector, ac, original_action, original_key)
            )

        # --- Phase 2: binary toggles ----------------------------------------
        # Runway availability
        for rwy, avail in sector.runway_availability.items():
            cf = self._try_runway_toggle(
                sector, rwy, avail, original_action, original_key
            )
            if cf:
                candidates.append(cf)

        # Emergency flag
        for ac in sector.aircraft:
            cf = self._try_emergency_toggle(
                sector, ac, original_action, original_key
            )
            if cf:
                candidates.append(cf)

        # Emergency type escalation / de-escalation
        for ac in sector.aircraft:
            cf = self._try_emergency_type(
                sector, ac, original_action, original_key
            )
            if cf:
                candidates.append(cf)

        if not candidates:
            return None

        return min(candidates, key=lambda c: c.normalized_cost)

    # ------------------------------------------------------------------
    # Perturbation probes
    # ------------------------------------------------------------------

    def _probe(
        self,
        perturbed: SectorState,
        original_key: str,
    ) -> Optional[CandidateAction]:
        """
        Run the recommender on `perturbed`.  Return the new action if it
        differs from the original, else None.
        """
        new_rec = self.recommender.recommend(perturbed)
        if _action_key(new_rec.action) != original_key:
            return new_rec.action
        return None

    def _fuel_crossings(
        self,
        sector: SectorState,
        ac: AircraftState,
        original_action: CandidateAction,
        original_key: str,
    ) -> list[Counterfactual]:
        """
        Compute the exact Δfuel needed to cross the is_fuel_critical boundary
        in either direction.  Verify each crossing with one recommender call.
        """
        burn    = ac.fuel_burn_rate_kg_per_min
        reserve = ac.reserve_fuel_kg
        if burn <= 0:
            return []

        results: list[Counterfactual] = []
        threshold_fuel = reserve + 30.0 * burn

        for target_minutes, label in [(30.1, "just-above"), (29.9, "just-below")]:
            target_fuel = reserve + target_minutes * burn
            delta       = target_fuel - ac.fuel_kg
            new_fuel    = ac.fuel_kg + delta

            # Realistic bounds: no negative fuel, no more than ×3 current
            if new_fuel < 0 or new_fuel > ac.fuel_kg * 3 + 1000:
                continue

            # Only probe if the crossing actually flips is_fuel_critical
            currently_critical = ac.is_fuel_critical
            would_be_critical  = new_fuel < threshold_fuel
            if currently_critical == would_be_critical:
                continue

            perturbed = _patch_aircraft(sector, ac.id, fuel_kg=new_fuel)
            new_action = self._probe(perturbed, original_key)
            if new_action is None:
                continue

            cost = abs(delta) / reserve
            results.append(Counterfactual(
                feature=f"{ac.id}.fuel_kg",
                original_value=ac.fuel_kg,
                perturbed_value=round(new_fuel, 1),
                normalized_cost=cost,
                original_action=original_key,
                counterfactual_action=_action_key(new_action),
                sentence=_sentence(
                    f"{ac.id}.fuel_kg", ac.fuel_kg, round(new_fuel, 1),
                    original_action, new_action, ac=ac,
                ),
            ))

        return results

    def _reserve_crossings(
        self,
        sector: SectorState,
        ac: AircraftState,
        original_action: CandidateAction,
        original_key: str,
    ) -> list[Counterfactual]:
        """
        Compute the exact Δreserve that crosses the is_fuel_critical boundary.
        """
        burn = ac.fuel_burn_rate_kg_per_min
        if burn <= 0:
            return []

        results: list[Counterfactual] = []

        # New reserve such that margin flips across 30-min boundary
        for target_minutes, label in [(30.1, "not-critical"), (29.9, "critical")]:
            # (fuel - new_reserve) / burn == target_minutes
            new_reserve = ac.fuel_kg - target_minutes * burn
            if new_reserve <= 0:
                continue

            delta = new_reserve - ac.reserve_fuel_kg
            currently_critical = ac.is_fuel_critical
            would_be_critical  = (ac.fuel_kg - new_reserve) / burn < 30.0
            if currently_critical == would_be_critical:
                continue

            # Realistic: reserve within [50%, 200%] of original
            if new_reserve < ac.reserve_fuel_kg * 0.5 or new_reserve > ac.reserve_fuel_kg * 2.0:
                continue

            perturbed = _patch_aircraft(sector, ac.id, reserve_fuel_kg=round(new_reserve, 1))
            new_action = self._probe(perturbed, original_key)
            if new_action is None:
                continue

            cost = abs(delta) / ac.reserve_fuel_kg
            results.append(Counterfactual(
                feature=f"{ac.id}.reserve_fuel_kg",
                original_value=ac.reserve_fuel_kg,
                perturbed_value=round(new_reserve, 1),
                normalized_cost=cost,
                original_action=original_key,
                counterfactual_action=_action_key(new_action),
                sentence=_sentence(
                    f"{ac.id}.reserve_fuel_kg", ac.reserve_fuel_kg,
                    round(new_reserve, 1), original_action, new_action, ac=ac,
                ),
            ))

        return results

    def _try_runway_toggle(
        self,
        sector: SectorState,
        rwy: str,
        current_avail: bool,
        original_action: CandidateAction,
        original_key: str,
    ) -> Optional[Counterfactual]:
        new_avail = not current_avail
        perturbed  = _patch_runway(sector, rwy, new_avail)
        new_action = self._probe(perturbed, original_key)
        if new_action is None:
            return None
        return Counterfactual(
            feature=f"runway_availability.{rwy}",
            original_value=current_avail,
            perturbed_value=new_avail,
            normalized_cost=1.0,
            original_action=original_key,
            counterfactual_action=_action_key(new_action),
            sentence=_sentence(
                f"runway_availability.{rwy}", current_avail, new_avail,
                original_action, new_action,
            ),
        )

    def _try_emergency_toggle(
        self,
        sector: SectorState,
        ac: AircraftState,
        original_action: CandidateAction,
        original_key: str,
    ) -> Optional[Counterfactual]:
        new_flag   = not ac.emergency_flag
        new_type   = "MAYDAY" if new_flag else None
        perturbed  = _patch_aircraft(
            sector, ac.id, emergency_flag=new_flag, emergency_type=new_type
        )
        new_action = self._probe(perturbed, original_key)
        if new_action is None:
            return None
        return Counterfactual(
            feature=f"{ac.id}.emergency_flag",
            original_value=ac.emergency_flag,
            perturbed_value=new_flag,
            normalized_cost=1.0,
            original_action=original_key,
            counterfactual_action=_action_key(new_action),
            sentence=_sentence(
                f"{ac.id}.emergency_flag", ac.emergency_flag, new_flag,
                original_action, new_action, ac=ac,
            ),
        )

    def _try_emergency_type(
        self,
        sector: SectorState,
        ac: AircraftState,
        original_action: CandidateAction,
        original_key: str,
    ) -> Optional[Counterfactual]:
        if not ac.emergency_flag:
            return None
        # Try escalating to MAYDAY (highest priority) or clearing
        for new_type in (["MAYDAY"] if ac.emergency_type != "MAYDAY" else [None]):
            new_flag  = new_type is not None
            perturbed = _patch_aircraft(
                sector, ac.id, emergency_flag=new_flag, emergency_type=new_type
            )
            new_action = self._probe(perturbed, original_key)
            if new_action is not None:
                return Counterfactual(
                    feature=f"{ac.id}.emergency_type",
                    original_value=ac.emergency_type,
                    perturbed_value=new_type,
                    normalized_cost=1.0,
                    original_action=original_key,
                    counterfactual_action=_action_key(new_action),
                    sentence=_sentence(
                        f"{ac.id}.emergency_type", ac.emergency_type, new_type,
                        original_action, new_action, ac=ac,
                    ),
                )
        return None


# ===========================================================================
# __main__ — toy 3-aircraft test
# ===========================================================================

if __name__ == "__main__":
    from state_schema import SectorState, AircraftState
    from bluesky_adapter import BlueSkyAdapter
    from cascade_engine import CascadeEngine
    from recommender import Recommender

    print("=== SKYLANCE-X Explainer — toy 3-aircraft test ===\n")

    # Toy sector design:
    #   AAA001 — fuel-critical only (NOT emergency); 25 min above reserve
    #             Recommender gives AssignRunwayAction(AAA001, 27L) for fuel_critical_bonus=80
    #   BBB002 — normal, ample fuel
    #   CCC003 — normal, ample fuel
    #
    # Expected recommendation: AssignRunwayAction(AAA001, 27L)
    #
    # Minimal flip: add (30.1 - 25) × 10 = 51 kg to AAA001's fuel
    #   → not fuel_critical → reward becomes 0 = NoAction's reward → NoAction wins
    #   Normalized cost = 51 / 2250 ≈ 0.023  (very small change)

    toy = SectorState(
        aircraft=[
            AircraftState(
                id="AAA001", lat=52.0, lon=4.0, altitude_ft=30000,
                heading_deg=90, ground_speed_kt=450,
                fuel_kg=2500.0,               # reserve + 25 min
                fuel_burn_rate_kg_per_min=10.0,
                reserve_fuel_kg=2250.0,       # threshold at 2250 + 300 = 2550 kg
                emergency_flag=False,         # NOT an emergency — pure fuel issue
                emergency_type=None,
                runway_needed="27L",
            ),
            AircraftState(
                id="BBB002", lat=52.5, lon=4.8, altitude_ft=32000,
                heading_deg=270, ground_speed_kt=420,
                fuel_kg=5000.0, fuel_burn_rate_kg_per_min=9.0,
                reserve_fuel_kg=2000.0,
                emergency_flag=False, emergency_type=None, runway_needed=None,
            ),
            AircraftState(
                id="CCC003", lat=51.7, lon=3.9, altitude_ft=34000,
                heading_deg=180, ground_speed_kt=430,
                fuel_kg=6000.0, fuel_burn_rate_kg_per_min=11.0,
                reserve_fuel_kg=2200.0,
                emergency_flag=False, emergency_type=None, runway_needed=None,
            ),
        ],
        runway_availability={"27L": True, "27R": True, "09L": True, "09R": True},
        sim_time_s=1800.0,
    )

    print("Sector:")
    for ac in toy.aircraft:
        margin = ac.fuel_minutes_above_reserve
        crit   = " [FUEL CRITICAL]" if ac.is_fuel_critical else ""
        emg    = f" [{ac.emergency_type}]" if ac.emergency_flag else ""
        print(f"  {ac.id}  FL{ac.altitude_ft/100:.0f}  "
              f"fuel={ac.fuel_kg:.0f}kg  reserve={ac.reserve_fuel_kg:.0f}kg  "
              f"margin={margin:.1f}min{crit}{emg}")

    print("\nInitialising BlueSky adapter...")
    adapter = BlueSkyAdapter()
    engine  = CascadeEngine(adapter, horizon_s=600.0, checkpoint_interval_s=60.0)
    rec     = Recommender(engine)
    exp     = Explainer(rec)

    print("\nRunning recommender on toy sector...")
    reco = rec.recommend(toy)
    print(f"  Recommendation: {_action_key(reco.action)}  "
          f"(reward={reco.reward:+.1f})")

    print("\nSearching for minimal-flip counterfactual...")
    cf = exp.explain(toy, reco)

    if cf is None:
        print("  No single-feature flip found within realistic bounds.")
    else:
        print(f"\n  Feature:           {cf.feature}")
        print(f"  Original value:    {cf.original_value}")
        print(f"  Perturbed value:   {cf.perturbed_value}")
        print(f"  Normalized cost:   {cf.normalized_cost:.4f}")
        print(f"  Original action:   {cf.original_action}")
        print(f"  New action:        {cf.counterfactual_action}")
        print(f"\n  Plain English:")
        print(f"  >>> {cf.sentence}")

    # ---- Also show ALL flips found (educational) ----
    print("\n--- All perturbations that cause a flip ---")
    original_key = _action_key(reco.action)
    all_cfs: list[Counterfactual] = []
    for ac in toy.aircraft:
        all_cfs.extend(exp._fuel_crossings(toy, ac, reco.action, original_key))
        all_cfs.extend(exp._reserve_crossings(toy, ac, reco.action, original_key))
        toggle = exp._try_emergency_toggle(toy, ac, reco.action, original_key)
        if toggle:
            all_cfs.append(toggle)
    for rwy, avail in toy.runway_availability.items():
        toggle = exp._try_runway_toggle(toy, rwy, avail, reco.action, original_key)
        if toggle:
            all_cfs.append(toggle)

    all_cfs.sort(key=lambda c: c.normalized_cost)
    for i, c in enumerate(all_cfs, 1):
        print(f"  [{i}] cost={c.normalized_cost:.4f}  {c.feature}: "
              f"{c.original_value!r} → {c.perturbed_value!r}  "
              f"→ {c.counterfactual_action}")
