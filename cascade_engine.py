"""
Cascade engine for SKYLANCE-X.

Takes a current SectorState and a candidate ATC action, projects the sector
forward in 60-second checkpoints using a pure-Python dead-reckoning simulator
(no BlueSky dependency), and scores two safety dimensions:
  1. Fuel breaches  — aircraft whose fuel_kg drops below reserve_fuel_kg
  2. Separation breaches — pairs that violate ICAO 5 nm / 1000 ft minima

Each breach carries the elapsed time and the aircraft involved, so the
suggestion layer can rank and explain candidates without further computation.

BlueSkyAdapter is NOT imported here.  The adapter is used only by the live
radar sim in the dashboard (bluesky_adapter.py / app.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

from state_schema import AircraftState, SectorState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_NM_TO_M     = 1852.0          # metres per nautical mile
_FT_TO_M     = 0.3048          # metres per foot
_EARTH_R_M   = 6_371_000.0     # mean Earth radius, metres
_NM_IN_DEG_LAT = 1.0 / 60.0   # 1 nm = 1/60 degree of latitude

# ICAO standard separation minima
_SEP_HORIZ_M = 5 * _NM_TO_M     # 9 260 m
_SEP_VERT_M  = 1000 * _FT_TO_M  # 304.8 m


# ===========================================================================
# Candidate actions
# ===========================================================================

@dataclass
class NoAction:
    """Baseline: controller makes no change.  Score this to establish a floor."""


@dataclass
class AssignRunwayAction:
    """Reassign an aircraft to a specific runway (priority landing request)."""
    aircraft_id: str
    runway_id: str          # e.g. "27L"


@dataclass
class HoldAction:
    """
    Vector an aircraft into a published hold.

    Modelled as a speed reduction to `hold_speed_kt`; heading is unchanged.
    The BlueSky propagation then moves the aircraft at that reduced speed,
    approximating the net displacement of an oval hold.
    """
    aircraft_id: str
    hold_speed_kt: float = 210.0   # typical narrowbody hold speed


@dataclass
class VectorAction:
    """Give an aircraft a new heading directive."""
    aircraft_id: str
    new_heading_deg: float         # degrees true, 0–360


CandidateAction = Union[NoAction, AssignRunwayAction, HoldAction, VectorAction]


# ===========================================================================
# Breach records — one per first detection per aircraft / pair
# ===========================================================================

@dataclass
class FuelBreach:
    aircraft_id: str
    sim_time_s: float     # absolute simulation time of first breach
    elapsed_s: float      # seconds after evaluation start
    fuel_kg: float        # fuel remaining at breach point
    reserve_kg: float     # threshold that was violated
    deficit_kg: float     # how far below reserve (positive = bad)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeparationBreach:
    aircraft_id_1: str
    aircraft_id_2: str
    sim_time_s: float
    elapsed_s: float
    horizontal_nm: float  # actual separation at breach
    vertical_ft: float    # actual vertical separation at breach

    def to_dict(self) -> dict:
        return asdict(self)


# ===========================================================================
# Score result
# ===========================================================================

@dataclass
class CascadeScore:
    """
    Complete scoring result for one candidate action over the lookahead horizon.

    Counts are de-duplicated: an aircraft that stays below reserve across
    multiple checkpoints counts once.  A pair that stays in conflict across
    multiple checkpoints counts once.
    """
    action: CandidateAction
    horizon_s: float              # evaluation window (seconds)
    checkpoint_interval_s: float  # granularity used

    fuel_breach_count: int
    separation_breach_count: int

    fuel_breaches: list[FuelBreach] = field(default_factory=list)
    separation_breaches: list[SeparationBreach] = field(default_factory=list)

    # First breach across both dimensions — key field for ranking
    time_to_first_breach_s: Optional[float] = None  # elapsed; None = clean
    first_breach_aircraft: Optional[str] = None      # callsign or "A / B"
    first_breach_type: Optional[str] = None          # "FUEL" or "SEPARATION"

    @property
    def is_safe(self) -> bool:
        return self.fuel_breach_count == 0 and self.separation_breach_count == 0

    @property
    def total_breach_count(self) -> int:
        return self.fuel_breach_count + self.separation_breach_count

    def to_dict(self) -> dict:
        d = {
            "action": asdict(self.action),
            "action_type": type(self.action).__name__,
            "horizon_s": self.horizon_s,
            "checkpoint_interval_s": self.checkpoint_interval_s,
            "fuel_breach_count": self.fuel_breach_count,
            "separation_breach_count": self.separation_breach_count,
            "fuel_breaches": [b.to_dict() for b in self.fuel_breaches],
            "separation_breaches": [b.to_dict() for b in self.separation_breaches],
            "time_to_first_breach_s": self.time_to_first_breach_s,
            "first_breach_aircraft": self.first_breach_aircraft,
            "first_breach_type": self.first_breach_type,
            "is_safe": self.is_safe,
        }
        return d


# ===========================================================================
# Engine
# ===========================================================================

def _advance_state(state: SectorState, dt_s: float) -> SectorState:
    """
    Dead-reckon all aircraft forward by dt_s seconds.

    Horizontal position:
        Each aircraft's airspeed (ground_speed_kt along heading) is resolved
        into north/east components, then the sector wind vector is added to
        give the actual ground-track displacement:

            track_north_kt = airspeed * cos(hdg) + wind_north_kt
            track_east_kt  = airspeed * sin(hdg) + wind_east_kt

        Positions are then updated using the flat-earth / cos(lat) correction.
        Error < 0.1 nm over a 10-min horizon at 450 kt — negligible against the
        5 nm separation minimum.  When wind is (0, 0) the formula is identical
        to the previous scalar-distance form.

    Altitude:
        Constant-rate model: altitude_ft += vertical_speed_fpm * (dt_s / 60),
        clamped to 0 ft.  The updated altitude feeds directly into
        _check_separation so the 1000 ft vertical test uses projected values.

    Wind:
        The wind vector is held constant throughout the lookahead and propagated
        unchanged into the returned SectorState.
    """
    dt_hr   = dt_s / 3600.0
    wind_n  = state.wind_north_kt
    wind_e  = state.wind_east_kt

    new_aircraft = []
    for ac in state.aircraft:
        hdg_rad = math.radians(ac.heading_deg)
        lat_rad = math.radians(ac.lat)

        track_n = ac.ground_speed_kt * math.cos(hdg_rad) + wind_n
        track_e = ac.ground_speed_kt * math.sin(hdg_rad) + wind_e
        cos_lat = math.cos(lat_rad)

        new_lat  = ac.lat + track_n * dt_hr * _NM_IN_DEG_LAT
        new_lon  = ac.lon + track_e * dt_hr * _NM_IN_DEG_LAT / max(cos_lat, 1e-9)
        new_fuel = max(0.0, ac.fuel_kg - ac.fuel_burn_rate_kg_per_min * (dt_s / 60.0))
        new_alt  = max(0.0, ac.altitude_ft + ac.vertical_speed_fpm * (dt_s / 60.0))

        new_aircraft.append(AircraftState(
            id=ac.id,
            lat=new_lat,
            lon=new_lon,
            altitude_ft=new_alt,
            heading_deg=ac.heading_deg,
            ground_speed_kt=ac.ground_speed_kt,
            fuel_kg=new_fuel,
            fuel_burn_rate_kg_per_min=ac.fuel_burn_rate_kg_per_min,
            reserve_fuel_kg=ac.reserve_fuel_kg,
            emergency_flag=ac.emergency_flag,
            emergency_type=ac.emergency_type,
            runway_needed=ac.runway_needed,
            vertical_speed_fpm=ac.vertical_speed_fpm,
        ))

    return SectorState(
        aircraft=new_aircraft,
        runway_availability=dict(state.runway_availability),
        sim_time_s=state.sim_time_s + dt_s,
        wind_north_kt=wind_n,
        wind_east_kt=wind_e,
    )


class CascadeEngine:
    """
    Anticipatory safety scorer for SKYLANCE-X.

    Pure-Python forward simulator — no BlueSky dependency.  Dead-reckons
    aircraft positions over the lookahead horizon, checks ICAO separation
    minima and fuel reserves at each checkpoint.  Evaluates in milliseconds.

    Typical usage:
        engine = CascadeEngine()
        baseline = engine.evaluate(sector, NoAction())
        score_a  = engine.evaluate(sector, AssignRunwayAction("KLM792", "27L"))
        score_b  = engine.evaluate(sector, HoldAction("AFR559"))
    """

    def __init__(
        self,
        _adapter=None,                    # accepted but not stored; kept for call-site compatibility
        horizon_s: float = 600.0,
        checkpoint_interval_s: float = 60.0,
    ) -> None:
        self.horizon_s = horizon_s
        self.checkpoint_interval_s = checkpoint_interval_s

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def evaluate(self, sector: SectorState, action: CandidateAction) -> CascadeScore:
        """
        Project `sector` forward by `horizon_s` after applying `action` and
        return a CascadeScore describing every safety breach encountered.

        Runs entirely in Python — no BlueSky calls, no I/O.
        """
        modified = _apply_action(sector, action)

        fuel_breaches: list[FuelBreach] = []
        sep_breaches: list[SeparationBreach] = []
        seen_fuel: set[str] = set()
        seen_sep: set[frozenset] = set()

        n_checkpoints = max(1, round(self.horizon_s / self.checkpoint_interval_s))
        state = modified

        for k in range(1, n_checkpoints + 1):
            state = _advance_state(state, self.checkpoint_interval_s)
            elapsed_s = k * self.checkpoint_interval_s

            # --- fuel check ---
            for ac in state.aircraft:
                if ac.id not in seen_fuel and ac.fuel_kg < ac.reserve_fuel_kg:
                    deficit = ac.reserve_fuel_kg - ac.fuel_kg
                    fuel_breaches.append(FuelBreach(
                        aircraft_id=ac.id,
                        sim_time_s=state.sim_time_s,
                        elapsed_s=elapsed_s,
                        fuel_kg=ac.fuel_kg,
                        reserve_kg=ac.reserve_fuel_kg,
                        deficit_kg=deficit,
                    ))
                    seen_fuel.add(ac.id)

            # --- separation check ---
            for breach in _check_separation(state):
                key = frozenset([breach.aircraft_id_1, breach.aircraft_id_2])
                if key not in seen_sep:
                    breach.sim_time_s = state.sim_time_s
                    breach.elapsed_s = elapsed_s
                    sep_breaches.append(breach)
                    seen_sep.add(key)

        # Determine first-breach summary
        first_fuel = min(fuel_breaches, key=lambda b: b.elapsed_s, default=None)
        first_sep  = min(sep_breaches,  key=lambda b: b.elapsed_s, default=None)

        time_to_first: Optional[float] = None
        first_aircraft: Optional[str] = None
        first_type: Optional[str] = None

        if first_fuel and first_sep:
            if first_fuel.elapsed_s <= first_sep.elapsed_s:
                time_to_first, first_aircraft, first_type = (
                    first_fuel.elapsed_s, first_fuel.aircraft_id, "FUEL"
                )
            else:
                time_to_first, first_aircraft, first_type = (
                    first_sep.elapsed_s,
                    f"{first_sep.aircraft_id_1} / {first_sep.aircraft_id_2}",
                    "SEPARATION",
                )
        elif first_fuel:
            time_to_first = first_fuel.elapsed_s
            first_aircraft = first_fuel.aircraft_id
            first_type = "FUEL"
        elif first_sep:
            time_to_first = first_sep.elapsed_s
            first_aircraft = f"{first_sep.aircraft_id_1} / {first_sep.aircraft_id_2}"
            first_type = "SEPARATION"

        return CascadeScore(
            action=action,
            horizon_s=self.horizon_s,
            checkpoint_interval_s=self.checkpoint_interval_s,
            fuel_breach_count=len(fuel_breaches),
            separation_breach_count=len(sep_breaches),
            fuel_breaches=fuel_breaches,
            separation_breaches=sep_breaches,
            time_to_first_breach_s=time_to_first,
            first_breach_aircraft=first_aircraft,
            first_breach_type=first_type,
        )

    def rank(
        self, sector: SectorState, actions: list[CandidateAction]
    ) -> list[tuple[CandidateAction, CascadeScore]]:
        """
        Evaluate every action in `actions` and return them sorted safest-first.

        Sort key: (total_breach_count, time_to_first_breach_s).
        An action with no breaches always ranks above one with breaches.
        Among equal breach counts, the action that delays the first breach
        longest ranks higher.
        """
        scored = [(a, self.evaluate(sector, a)) for a in actions]
        scored.sort(key=lambda pair: (
            pair[1].total_breach_count,
            pair[1].time_to_first_breach_s if pair[1].time_to_first_breach_s is not None
            else float("inf"),
        ))
        return scored


# ===========================================================================
# Action application  (pure — returns a new SectorState, never mutates input)
# ===========================================================================

def _apply_action(sector: SectorState, action: CandidateAction) -> SectorState:
    """
    Return a new SectorState reflecting the controller action.

    Only the fields directly altered by the action change; everything else is
    a shallow copy.  The modification is applied to the schema, then the
    adapter propagates it via load_scenario → BlueSky.
    """
    aircraft = [_copy_ac(ac) for ac in sector.aircraft]
    runway_availability = dict(sector.runway_availability)

    if isinstance(action, NoAction):
        pass  # nothing to change

    elif isinstance(action, AssignRunwayAction):
        for ac in aircraft:
            if ac.id == action.aircraft_id:
                ac.runway_needed = action.runway_id
                break

    elif isinstance(action, HoldAction):
        for ac in aircraft:
            if ac.id == action.aircraft_id:
                # Clamp to [min cruise speed, current speed]
                ac.ground_speed_kt = min(action.hold_speed_kt, ac.ground_speed_kt)
                break

    elif isinstance(action, VectorAction):
        for ac in aircraft:
            if ac.id == action.aircraft_id:
                ac.heading_deg = action.new_heading_deg % 360.0
                break

    else:
        raise TypeError(f"Unknown action type: {type(action)}")

    return SectorState(
        aircraft=aircraft,
        runway_availability=runway_availability,
        sim_time_s=sector.sim_time_s,
        wind_north_kt=sector.wind_north_kt,
        wind_east_kt=sector.wind_east_kt,
    )


def _copy_ac(ac: AircraftState) -> AircraftState:
    """Shallow-copy an AircraftState (all fields are primitives, so this is a deep copy)."""
    return AircraftState(
        id=ac.id,
        lat=ac.lat,
        lon=ac.lon,
        altitude_ft=ac.altitude_ft,
        heading_deg=ac.heading_deg,
        ground_speed_kt=ac.ground_speed_kt,
        fuel_kg=ac.fuel_kg,
        fuel_burn_rate_kg_per_min=ac.fuel_burn_rate_kg_per_min,
        reserve_fuel_kg=ac.reserve_fuel_kg,
        emergency_flag=ac.emergency_flag,
        emergency_type=ac.emergency_type,
        runway_needed=ac.runway_needed,
        vertical_speed_fpm=ac.vertical_speed_fpm,
    )


# ===========================================================================
# ICAO separation check  (geometry only — no BlueSky dependency)
# ===========================================================================

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_R_M * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _check_separation(state: SectorState) -> list[SeparationBreach]:
    """
    Return one SeparationBreach per violating pair in `state`.

    Both the horizontal AND vertical minima must be breached simultaneously
    for a pair to count (ICAO Doc 4444 standard).

    sim_time_s and elapsed_s are left at 0 — the caller fills them in.
    """
    breaches: list[SeparationBreach] = []
    ac = state.aircraft
    n = len(ac)

    for i in range(n):
        for j in range(i + 1, n):
            horiz_m = _haversine_m(ac[i].lat, ac[i].lon, ac[j].lat, ac[j].lon)
            vert_m  = abs(ac[i].altitude_ft - ac[j].altitude_ft) * _FT_TO_M

            if horiz_m < _SEP_HORIZ_M and vert_m < _SEP_VERT_M:
                breaches.append(SeparationBreach(
                    aircraft_id_1=ac[i].id,
                    aircraft_id_2=ac[j].id,
                    sim_time_s=0.0,    # filled by caller
                    elapsed_s=0.0,     # filled by caller
                    horizontal_nm=horiz_m / _NM_TO_M,
                    vertical_ft=vert_m / _FT_TO_M,
                ))

    return breaches


# ===========================================================================
# __main__ smoke-test
# ===========================================================================

if __name__ == "__main__":
    import json
    from state_schema import make_mock_sector

    print("=== SKYLANCE-X Cascade Engine smoke-test ===\n")

    sector = make_mock_sector(n=12, seed=42)
    engine  = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)

    # Print initial critical aircraft
    print("Initial sector summary:")
    for ac in sector.aircraft:
        mins = ac.fuel_minutes_above_reserve
        tag  = f"  *** FUEL CRITICAL ({mins:.0f} min above reserve)" if ac.is_fuel_critical else ""
        emg  = f"  [{ac.emergency_type}]" if ac.emergency_flag else ""
        print(f"  {ac.id:<8}  FL{ac.altitude_ft/100:03.0f}  "
              f"fuel={ac.fuel_kg:.0f}kg  reserve={ac.reserve_fuel_kg:.0f}kg  "
              f"burn={ac.fuel_burn_rate_kg_per_min:.1f}kg/min{emg}{tag}")

    print()
    fuel_critical = [ac for ac in sector.aircraft if ac.is_fuel_critical]
    emergency_ac  = [ac for ac in sector.aircraft if ac.emergency_flag]

    # Build a set of actions to compare
    actions: list[CandidateAction] = [NoAction()]

    # Runway assignment for fuel-critical aircraft
    if fuel_critical:
        ac = fuel_critical[0]
        for rwy, available in sector.runway_availability.items():
            if available:
                actions.append(AssignRunwayAction(ac.id, rwy))
                break

    # Hold action for an aircraft with the highest burn rate
    busiest = max(sector.aircraft, key=lambda a: a.fuel_burn_rate_kg_per_min)
    actions.append(HoldAction(busiest.id, hold_speed_kt=210.0))

    # Vector action for the first emergency aircraft
    if emergency_ac:
        actions.append(VectorAction(emergency_ac[0].id, new_heading_deg=180.0))

    print(f"Evaluating {len(actions)} actions over 10-minute horizon...\n")

    ranked = engine.rank(sector, actions)

    for rank_pos, (action, score) in enumerate(ranked, 1):
        action_desc = (
            f"NoAction"
            if isinstance(action, NoAction)
            else f"{type(action).__name__}({getattr(action, 'aircraft_id', '')} "
                 f"{getattr(action, 'runway_id', getattr(action, 'new_heading_deg', getattr(action, 'hold_speed_kt', '')))})"
        )
        breach_summary = (
            "SAFE — no breaches"
            if score.is_safe
            else (
                f"{score.fuel_breach_count} fuel breach(es), "
                f"{score.separation_breach_count} sep breach(es), "
                f"first at t+{score.time_to_first_breach_s:.0f}s "
                f"({score.first_breach_type}: {score.first_breach_aircraft})"
            )
        )
        print(f"  #{rank_pos}  {action_desc:<55}  {breach_summary}")

        if score.fuel_breaches:
            for b in score.fuel_breaches:
                print(f"       FUEL  {b.aircraft_id:<8}  t+{b.elapsed_s:.0f}s  "
                      f"fuel={b.fuel_kg:.0f}kg < reserve={b.reserve_kg:.0f}kg  "
                      f"deficit={b.deficit_kg:.0f}kg")
        if score.separation_breaches:
            for b in score.separation_breaches:
                print(f"       SEP   {b.aircraft_id_1}/{b.aircraft_id_2}  "
                      f"t+{b.elapsed_s:.0f}s  "
                      f"horiz={b.horizontal_nm:.2f}nm  vert={b.vertical_ft:.0f}ft")

    print("\nJSON output for best action:")
    best_score = ranked[0][1]
    print(json.dumps(best_score.to_dict(), indent=2))
