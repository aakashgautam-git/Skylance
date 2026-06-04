"""
Counterfactual minimal-perturbation explainer for SKYLANCE-X.

Given the chosen action and the current SectorState, finds the smallest-cost
feature change (or feature pair) that flips the recommendation, and returns it
as a plain-English sentence.

Search strategy
---------------
Phase 1 — Single-feature analytical crossings (cheapest; exact delta, no search):
  Compute the exact Δfuel / Δreserve needed to cross the is_fuel_critical
  30-minute threshold, then verify with one recommender call per candidate.

Phase 2 — Single-feature binary toggles:
  Flip each runway, emergency_flag, and emergency_type.
  One recommender call per candidate.

Phase 3 — Two-feature pair search (runs only when Phase 1+2 find nothing):
  Collect all valid Phase-1/2 perturbation atoms, form pairs sorted by combined
  normalized cost, and probe up to `pair_budget` pairs.  Skips same-feature
  pairs (second would override first) and conflicting emergency pairs on the
  same aircraft.  Default budget = 50 pairs (~5–30 s depending on sector size).

Normalized cost
---------------
  fuel_kg change:         |Δkg| / reserve_fuel_kg
  reserve_fuel_kg change: |Δreserve| / reserve_fuel_kg
  binary toggle:          1.0
  pair:                   cost_a + cost_b
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from state_schema import AircraftState, SectorState
from cascade_engine import (
    CandidateAction,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
)
from recommender import Recommender, Recommendation

_THRESHOLD_EPS  = 0.5    # kg — nudge past threshold boundary
_HOLD_SPEED_MIN = 180.0  # kt — floor for realistic holding speeds


# ===========================================================================
# Action identity helpers
# ===========================================================================

def _action_key(action: CandidateAction) -> str:
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
# Perturbation atom  (internal — used by pair search)
# ===========================================================================

class _Perturbation:
    """
    One candidate feature change, stored as data + a pure transform.

    `apply` is a function SectorState → SectorState that applies this
    perturbation to any base sector, making pairs composable:

        combined = pert_b.apply(pert_a.apply(base_sector))

    Both `pert_a` and `pert_b` capture their own overrides in default
    arguments so the lambdas are safe to use after the loop that created them.
    """
    __slots__ = ("feature", "original_value", "perturbed_value", "cost", "apply", "ac")

    def __init__(
        self,
        feature: str,
        original_value: Any,
        perturbed_value: Any,
        cost: float,
        apply: Callable[[SectorState], SectorState],
        ac: Optional[AircraftState] = None,
    ) -> None:
        self.feature        = feature
        self.original_value = original_value
        self.perturbed_value = perturbed_value
        self.cost           = cost
        self.apply          = apply
        self.ac             = ac


# ===========================================================================
# Sector patch helpers  (pure)
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
        "vertical_speed_fpm": ac.vertical_speed_fpm,
    }
    d.update(overrides)
    return AircraftState(**d)


def _patch_aircraft(sector: SectorState, aircraft_id: str, **overrides) -> SectorState:
    return SectorState(
        aircraft=[
            _copy_aircraft(ac, **overrides) if ac.id == aircraft_id else ac
            for ac in sector.aircraft
        ],
        runway_availability=dict(sector.runway_availability),
        sim_time_s=sector.sim_time_s,
        wind_north_kt=sector.wind_north_kt,
        wind_east_kt=sector.wind_east_kt,
    )


def _patch_runway(sector: SectorState, runway_id: str, available: bool) -> SectorState:
    rwa = dict(sector.runway_availability)
    rwa[runway_id] = available
    return SectorState(
        aircraft=list(sector.aircraft),
        runway_availability=rwa,
        sim_time_s=sector.sim_time_s,
        wind_north_kt=sector.wind_north_kt,
        wind_east_kt=sector.wind_east_kt,
    )


# ===========================================================================
# Counterfactual record
# ===========================================================================

@dataclass
class Counterfactual:
    feature: str               # "KLM792.fuel_kg"  or  "A.fuel_kg + runway.27L"
    original_value: Any        # scalar  or  2-tuple for pairs
    perturbed_value: Any       # scalar  or  2-tuple for pairs
    normalized_cost: float     # |Δ|/scale  or  cost_a + cost_b for pairs
    original_action: str       # key of original recommendation
    counterfactual_action: str # key of recommendation after perturbation
    sentence: str              # plain-English explanation


# ===========================================================================
# Sentence helpers
# ===========================================================================

def _condition_clause(
    feature: str,
    original_value: Any,
    perturbed_value: Any,
    ac: Optional[AircraftState] = None,
) -> str:
    """
    Condition phrase for use in  "If <clause>, the system would …"

    Returns the clause without a leading "If" and without the action suffix,
    so it can be combined for pair explanations:
        "If <clause_a> and <clause_b>, the system would …"
    """
    if feature.endswith(".fuel_kg") and ac is not None:
        delta = perturbed_value - original_value
        sign  = "increased" if delta > 0 else "decreased"
        burn  = ac.fuel_burn_rate_kg_per_min
        from_margin = ac.fuel_minutes_above_reserve
        to_margin   = (perturbed_value - ac.reserve_fuel_kg) / burn if burn > 0 else 0
        return (
            f"{ac.id}'s fuel {sign} by {abs(delta):.0f} kg "
            f"(from {original_value:.0f} kg to {perturbed_value:.0f} kg, "
            f"margin {from_margin:.1f} → {to_margin:.1f} min above reserve)"
        )

    if feature.endswith(".reserve_fuel_kg") and ac is not None:
        delta = perturbed_value - original_value
        sign  = "increased" if delta > 0 else "decreased"
        return (
            f"{ac.id}'s regulatory reserve {sign} by {abs(delta):.0f} kg "
            f"(from {original_value:.0f} kg to {perturbed_value:.0f} kg)"
        )

    if feature.endswith(".emergency_flag") and ac is not None:
        return (
            f"{ac.id} declared an emergency"
            if perturbed_value
            else f"{ac.id}'s emergency were resolved"
        )

    if feature.endswith(".emergency_type") and ac is not None:
        if perturbed_value is None:
            return f"{ac.id}'s emergency type were cleared"
        return f"{ac.id}'s emergency were upgraded to {perturbed_value}"

    if "runway_availability" in feature:
        rwy   = feature.split(".")[-1]
        state = "closed" if not perturbed_value else "opened"
        return f"runway {rwy} were {state}"

    return f"{feature} changed from {original_value!r} to {perturbed_value!r}"


def _sentence(
    feature: str,
    original_value: Any,
    perturbed_value: Any,
    original_action: CandidateAction,
    new_action: CandidateAction,
    ac: Optional[AircraftState] = None,
) -> str:
    clause = _condition_clause(feature, original_value, perturbed_value, ac)
    return (
        f"If {clause}, "
        f"the system would {_action_label(new_action)} "
        f"instead of {_action_label(original_action)}."
    )


def _pair_sentence(
    pert_a: _Perturbation,
    pert_b: _Perturbation,
    original_action: CandidateAction,
    new_action: CandidateAction,
) -> str:
    """Plain-English sentence for a two-feature counterfactual."""
    clause_a = _condition_clause(
        pert_a.feature, pert_a.original_value, pert_a.perturbed_value, pert_a.ac,
    )
    clause_b = _condition_clause(
        pert_b.feature, pert_b.original_value, pert_b.perturbed_value, pert_b.ac,
    )
    return (
        f"If {clause_a} and {clause_b}, "
        f"the system would {_action_label(new_action)} "
        f"instead of {_action_label(original_action)}."
    )


# ===========================================================================
# Core explainer
# ===========================================================================

class Explainer:
    """
    Finds the minimal-cost single-feature perturbation — or, if none exists,
    the minimal-cost two-feature pair — that flips the recommendation.

    Usage:
        exp = Explainer(recommender)
        cf  = exp.explain(sector, recommendation)
        if cf:
            print(cf.sentence)

    pair_budget controls how many pairs Phase 3 probes (default 50).
    Set pair_budget=0 to restrict to single-feature only.
    """

    def __init__(self, recommender: Recommender) -> None:
        self.recommender = recommender

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        sector: SectorState,
        recommendation: Recommendation,
        pair_budget: int = 50,
    ) -> Optional[Counterfactual]:
        """
        Return the cheapest Counterfactual found, or None.

        Phases 1+2 search single-feature perturbations and return immediately
        when a flip is found.  Phase 3 (pair search) runs only when Phases 1+2
        find nothing, and is capped at `pair_budget` recommender calls.
        """
        original_action = recommendation.action
        original_key    = _action_key(original_action)

        # --- Phase 1: analytical fuel / reserve crossings -------------------
        candidates: list[Counterfactual] = []
        for ac in sector.aircraft:
            candidates.extend(self._fuel_crossings(sector, ac, original_action, original_key))
            candidates.extend(self._reserve_crossings(sector, ac, original_action, original_key))

        # --- Phase 2: binary toggles ----------------------------------------
        for rwy, avail in sector.runway_availability.items():
            cf = self._try_runway_toggle(sector, rwy, avail, original_action, original_key)
            if cf:
                candidates.append(cf)
        for ac in sector.aircraft:
            cf = self._try_emergency_toggle(sector, ac, original_action, original_key)
            if cf:
                candidates.append(cf)
        for ac in sector.aircraft:
            cf = self._try_emergency_type(sector, ac, original_action, original_key)
            if cf:
                candidates.append(cf)

        if candidates:
            return min(candidates, key=lambda c: c.normalized_cost)

        # --- Phase 3: pair search -------------------------------------------
        if pair_budget <= 0:
            return None

        return self._pair_search(sector, original_action, original_key, pair_budget)

    # ------------------------------------------------------------------
    # Atom generators  (produce _Perturbation objects without probing)
    # ------------------------------------------------------------------

    def _fuel_perturbation_atoms(self, ac: AircraftState) -> list[_Perturbation]:
        """Fuel-crossing atoms: exact Δfuel to cross the is_fuel_critical boundary."""
        burn    = ac.fuel_burn_rate_kg_per_min
        reserve = ac.reserve_fuel_kg
        if burn <= 0:
            return []

        atoms: list[_Perturbation] = []
        threshold_fuel = reserve + 30.0 * burn

        for target_minutes in (30.1, 29.9):
            target_fuel = reserve + target_minutes * burn
            delta       = target_fuel - ac.fuel_kg
            new_fuel    = ac.fuel_kg + delta

            if new_fuel < 0 or new_fuel > ac.fuel_kg * 3 + 1000:
                continue
            currently_critical = ac.is_fuel_critical
            would_be_critical  = new_fuel < threshold_fuel
            if currently_critical == would_be_critical:
                continue

            new_fuel_r = round(new_fuel, 1)
            atoms.append(_Perturbation(
                feature=f"{ac.id}.fuel_kg",
                original_value=ac.fuel_kg,
                perturbed_value=new_fuel_r,
                cost=abs(delta) / reserve,
                apply=lambda s, _id=ac.id, _nf=new_fuel_r: _patch_aircraft(s, _id, fuel_kg=_nf),
                ac=ac,
            ))
        return atoms

    def _reserve_perturbation_atoms(self, ac: AircraftState) -> list[_Perturbation]:
        """Reserve-crossing atoms: exact Δreserve to cross the is_fuel_critical boundary."""
        burn = ac.fuel_burn_rate_kg_per_min
        if burn <= 0:
            return []

        atoms: list[_Perturbation] = []
        for target_minutes in (30.1, 29.9):
            new_reserve = ac.fuel_kg - target_minutes * burn
            if new_reserve <= 0:
                continue
            delta = new_reserve - ac.reserve_fuel_kg
            if ac.is_fuel_critical == ((ac.fuel_kg - new_reserve) / burn < 30.0):
                continue
            if new_reserve < ac.reserve_fuel_kg * 0.5 or new_reserve > ac.reserve_fuel_kg * 2.0:
                continue

            new_reserve_r = round(new_reserve, 1)
            atoms.append(_Perturbation(
                feature=f"{ac.id}.reserve_fuel_kg",
                original_value=ac.reserve_fuel_kg,
                perturbed_value=new_reserve_r,
                cost=abs(delta) / ac.reserve_fuel_kg,
                apply=lambda s, _id=ac.id, _nr=new_reserve_r: _patch_aircraft(s, _id, reserve_fuel_kg=_nr),
                ac=ac,
            ))
        return atoms

    def _runway_toggle_atom(self, rwy: str, avail: bool) -> _Perturbation:
        new_avail = not avail
        return _Perturbation(
            feature=f"runway_availability.{rwy}",
            original_value=avail,
            perturbed_value=new_avail,
            cost=1.0,
            apply=lambda s, _r=rwy, _a=new_avail: _patch_runway(s, _r, _a),
        )

    def _emergency_toggle_atom(self, ac: AircraftState) -> _Perturbation:
        new_flag = not ac.emergency_flag
        new_type = "MAYDAY" if new_flag else None
        return _Perturbation(
            feature=f"{ac.id}.emergency_flag",
            original_value=ac.emergency_flag,
            perturbed_value=new_flag,
            cost=1.0,
            apply=lambda s, _id=ac.id, _nf=new_flag, _nt=new_type: (
                _patch_aircraft(s, _id, emergency_flag=_nf, emergency_type=_nt)
            ),
            ac=ac,
        )

    def _emergency_type_atom(self, ac: AircraftState) -> Optional[_Perturbation]:
        if not ac.emergency_flag:
            return None
        new_type = "MAYDAY" if ac.emergency_type != "MAYDAY" else None
        new_flag = new_type is not None
        return _Perturbation(
            feature=f"{ac.id}.emergency_type",
            original_value=ac.emergency_type,
            perturbed_value=new_type,
            cost=1.0,
            apply=lambda s, _id=ac.id, _nf=new_flag, _nt=new_type: (
                _patch_aircraft(s, _id, emergency_flag=_nf, emergency_type=_nt)
            ),
            ac=ac,
        )

    # ------------------------------------------------------------------
    # Probe helpers
    # ------------------------------------------------------------------

    def _probe(self, perturbed: SectorState, original_key: str) -> Optional[CandidateAction]:
        new_rec = self.recommender.recommend(perturbed)
        if _action_key(new_rec.action) != original_key:
            return new_rec.action
        return None

    def _probe_atom(
        self,
        sector: SectorState,
        atom: _Perturbation,
        original_action: CandidateAction,
        original_key: str,
    ) -> Optional[Counterfactual]:
        """Apply one atom to sector, probe the recommender, return CF or None."""
        new_action = self._probe(atom.apply(sector), original_key)
        if new_action is None:
            return None
        return Counterfactual(
            feature=atom.feature,
            original_value=atom.original_value,
            perturbed_value=atom.perturbed_value,
            normalized_cost=atom.cost,
            original_action=original_key,
            counterfactual_action=_action_key(new_action),
            sentence=_sentence(
                atom.feature, atom.original_value, atom.perturbed_value,
                original_action, new_action, ac=atom.ac,
            ),
        )

    # ------------------------------------------------------------------
    # Phase 1+2 probe methods (public, keep original signatures for tests)
    # ------------------------------------------------------------------

    def _fuel_crossings(
        self, sector: SectorState, ac: AircraftState,
        original_action: CandidateAction, original_key: str,
    ) -> list[Counterfactual]:
        return [
            cf for cf in (
                self._probe_atom(sector, atom, original_action, original_key)
                for atom in self._fuel_perturbation_atoms(ac)
            )
            if cf is not None
        ]

    def _reserve_crossings(
        self, sector: SectorState, ac: AircraftState,
        original_action: CandidateAction, original_key: str,
    ) -> list[Counterfactual]:
        return [
            cf for cf in (
                self._probe_atom(sector, atom, original_action, original_key)
                for atom in self._reserve_perturbation_atoms(ac)
            )
            if cf is not None
        ]

    def _try_runway_toggle(
        self, sector: SectorState, rwy: str, avail: bool,
        original_action: CandidateAction, original_key: str,
    ) -> Optional[Counterfactual]:
        return self._probe_atom(
            sector, self._runway_toggle_atom(rwy, avail), original_action, original_key,
        )

    def _try_emergency_toggle(
        self, sector: SectorState, ac: AircraftState,
        original_action: CandidateAction, original_key: str,
    ) -> Optional[Counterfactual]:
        return self._probe_atom(
            sector, self._emergency_toggle_atom(ac), original_action, original_key,
        )

    def _try_emergency_type(
        self, sector: SectorState, ac: AircraftState,
        original_action: CandidateAction, original_key: str,
    ) -> Optional[Counterfactual]:
        atom = self._emergency_type_atom(ac)
        if atom is None:
            return None
        return self._probe_atom(sector, atom, original_action, original_key)

    # ------------------------------------------------------------------
    # Phase 3: pair search
    # ------------------------------------------------------------------

    def _collect_perturbations(self, sector: SectorState) -> list[_Perturbation]:
        """
        Assemble all single-feature perturbation atoms — the same candidates
        Phases 1+2 probe, returned as data for pair composition.
        """
        atoms: list[_Perturbation] = []
        for ac in sector.aircraft:
            atoms.extend(self._fuel_perturbation_atoms(ac))
            atoms.extend(self._reserve_perturbation_atoms(ac))
        for rwy, avail in sector.runway_availability.items():
            atoms.append(self._runway_toggle_atom(rwy, avail))
        for ac in sector.aircraft:
            atoms.append(self._emergency_toggle_atom(ac))
            emg_type = self._emergency_type_atom(ac)
            if emg_type is not None:
                atoms.append(emg_type)
        return atoms

    @staticmethod
    def _features_conflict(fa: str, fb: str) -> bool:
        """
        True when applying both perturbations to the same sector is undefined
        or contradictory.

        Two features conflict when:
        - They are the same field (second would silently override the first).
        - They both modify emergency_flag/emergency_type on the same aircraft
          (e.g. toggle + type change — the two applies contradict each other).
        """
        if fa == fb:
            return True
        emg_fields = (".emergency_flag", ".emergency_type")
        if any(fa.endswith(f) for f in emg_fields) and any(fb.endswith(f) for f in emg_fields):
            ac_a = fa.split(".")[0]
            ac_b = fb.split(".")[0]
            if ac_a == ac_b:
                return True
        return False

    def _pair_search(
        self,
        sector: SectorState,
        original_action: CandidateAction,
        original_key: str,
        pair_budget: int,
    ) -> Optional[Counterfactual]:
        """
        Try pairs of single-feature perturbations, cheapest-first.

        Pairs are sorted by combined normalized cost so the most plausible
        explanations are found first.  Stops after `pair_budget` probes.
        Returns the cheapest pair counterfactual found, or None.
        """
        atoms = self._collect_perturbations(sector)
        n     = len(atoms)

        # Build sorted list of valid (i, j) pairs
        pairs: list[tuple[_Perturbation, _Perturbation]] = sorted(
            (
                (atoms[i], atoms[j])
                for i in range(n)
                for j in range(i + 1, n)
                if not self._features_conflict(atoms[i].feature, atoms[j].feature)
            ),
            key=lambda p: p[0].cost + p[1].cost,
        )

        best: Optional[Counterfactual] = None
        probed = 0

        for a, b in pairs:
            if probed >= pair_budget:
                break
            combined   = b.apply(a.apply(sector))
            new_action = self._probe(combined, original_key)
            probed    += 1

            if new_action is None:
                continue

            cf = Counterfactual(
                feature=f"{a.feature} + {b.feature}",
                original_value=(a.original_value, b.original_value),
                perturbed_value=(a.perturbed_value, b.perturbed_value),
                normalized_cost=a.cost + b.cost,
                original_action=original_key,
                counterfactual_action=_action_key(new_action),
                sentence=_pair_sentence(a, b, original_action, new_action),
            )
            # Keep cheapest; stop early if we find a globally-minimal pair
            # (combined cost ≤ smallest possible remaining = current pair cost,
            # since list is sorted — first hit is already optimal).
            if best is None or cf.normalized_cost < best.normalized_cost:
                best = cf
            break  # sorted order means first hit is the cheapest pair

        return best


# ===========================================================================
# __main__ — smoke-test
# ===========================================================================

if __name__ == "__main__":
    from state_schema import SectorState, AircraftState
    from bluesky_adapter import BlueSkyAdapter
    from cascade_engine import CascadeEngine
    from recommender import Recommender

    print("=== SKYLANCE-X Explainer smoke-test ===\n")

    # ── Toy sector (single-feature flip expected) ─────────────────────────────
    # AAA001 — fuel-critical only, 25 min above reserve → AssignRunway(AAA001, 27L)
    # Minimal flip: add ~51 kg fuel → not critical → NoAction wins
    toy = SectorState(
        aircraft=[
            AircraftState(
                id="AAA001", lat=52.0, lon=4.0, altitude_ft=30000,
                heading_deg=90, ground_speed_kt=450,
                fuel_kg=2500.0, fuel_burn_rate_kg_per_min=10.0,
                reserve_fuel_kg=2250.0,
                emergency_flag=False, emergency_type=None, runway_needed="27L",
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
        tag    = " [FUEL CRITICAL]" if ac.is_fuel_critical else ""
        print(f"  {ac.id}  FL{ac.altitude_ft/100:.0f}  "
              f"fuel={ac.fuel_kg:.0f}kg  reserve={ac.reserve_fuel_kg:.0f}kg  "
              f"margin={margin:.1f}min{tag}")

    print("\nInitialising BlueSky adapter…")
    adapter = BlueSkyAdapter()
    engine  = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)
    rec     = Recommender(engine)
    exp     = Explainer(rec)

    print("\nRecommendation:")
    reco = rec.recommend(toy)
    print(f"  {_action_key(reco.action)}  (reward={reco.reward:+.1f})")

    # ── Single-feature explanation ────────────────────────────────────────────
    print("\nPhase 1+2 (single-feature, pair_budget=0):")
    cf_single = exp.explain(toy, reco, pair_budget=0)
    if cf_single:
        print(f"  cost={cf_single.normalized_cost:.4f}  {cf_single.feature}")
        print(f"  >>> {cf_single.sentence}")
    else:
        print("  No single-feature flip found.")

    # ── With pair search enabled (default) ───────────────────────────────────
    print("\nPhase 1+2+3 (pair_budget=50):")
    cf_pair = exp.explain(toy, reco, pair_budget=50)
    if cf_pair:
        print(f"  cost={cf_pair.normalized_cost:.4f}  {cf_pair.feature}")
        print(f"  >>> {cf_pair.sentence}")
    else:
        print("  No flip found within budget.")

    # ── Demonstrate pair search on a harder sector ───────────────────────────
    # Two fuel-critical aircraft; two open runways.
    # Single-feature: closing one runway just redirects to the other runway
    #   → still AssignRunway (different runway), so it IS a single-feature flip.
    # To reach NoAction both runways must close simultaneously (pair).
    # This exercises the pair search path explicitly.
    print("\n── Pair-search demonstration sector ──")
    hard = SectorState(
        aircraft=[
            AircraftState(
                id="EMG001", lat=52.0, lon=4.0, altitude_ft=30000,
                heading_deg=90, ground_speed_kt=450,
                fuel_kg=2200.0, fuel_burn_rate_kg_per_min=10.0,
                reserve_fuel_kg=2000.0,
                emergency_flag=True, emergency_type="MAYDAY", runway_needed="27L",
            ),
            AircraftState(
                id="NRM001", lat=52.5, lon=4.5, altitude_ft=32000,
                heading_deg=180, ground_speed_kt=430,
                fuel_kg=8000.0, fuel_burn_rate_kg_per_min=10.0,
                reserve_fuel_kg=2000.0,
                emergency_flag=False, emergency_type=None, runway_needed=None,
            ),
        ],
        runway_availability={"27L": True, "27R": True},
        sim_time_s=0.0,
    )
    reco_hard = rec.recommend(hard)
    print(f"Recommendation: {_action_key(reco_hard.action)}")

    cf_hard_single = exp.explain(hard, reco_hard, pair_budget=0)
    cf_hard_pair   = exp.explain(hard, reco_hard, pair_budget=50)

    print(f"Single-feature: "
          f"{cf_hard_single.feature if cf_hard_single else 'none'}")
    if cf_hard_single:
        print(f"  >>> {cf_hard_single.sentence}")

    print(f"With pairs:     "
          f"{cf_hard_pair.feature if cf_hard_pair else 'none'}")
    if cf_hard_pair:
        print(f"  >>> {cf_hard_pair.sentence}")

    # ── All single-feature candidates (educational) ──────────────────────────
    print("\n--- All single-feature perturbations that flip (toy sector) ---")
    orig_key = _action_key(reco.action)
    all_cfs: list[Counterfactual] = []
    for ac in toy.aircraft:
        all_cfs.extend(exp._fuel_crossings(toy, ac, reco.action, orig_key))
        all_cfs.extend(exp._reserve_crossings(toy, ac, reco.action, orig_key))
        t = exp._try_emergency_toggle(toy, ac, reco.action, orig_key)
        if t:
            all_cfs.append(t)
    for rwy, avail in toy.runway_availability.items():
        t = exp._try_runway_toggle(toy, rwy, avail, reco.action, orig_key)
        if t:
            all_cfs.append(t)
    all_cfs.sort(key=lambda c: c.normalized_cost)
    for i, c in enumerate(all_cfs, 1):
        print(f"  [{i}] cost={c.normalized_cost:.4f}  {c.feature}: "
              f"{c.original_value!r} → {c.perturbed_value!r}  → {c.counterfactual_action}")
