"""
Canonical sector state schema for SKYLANCE-X.

This file is the single source of truth for data structures shared across all
modules (BlueSky adapter, suggestion engine, UI, tests). Do not change field
names without versioning — every other module mocks against this schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AircraftState:
    id: str                          # ICAO callsign, e.g. "BAW123"
    lat: float                       # degrees, WGS-84
    lon: float                       # degrees, WGS-84
    altitude_ft: float               # feet MSL
    heading_deg: float               # degrees true, 0–360
    ground_speed_kt: float           # knots
    fuel_kg: float                   # current usable fuel, kg
    fuel_burn_rate_kg_per_min: float # current burn rate, kg/min
    reserve_fuel_kg: float           # minimum reserve required by regulation, kg
    emergency_flag: bool             # True if any emergency is active
    emergency_type: Optional[str]    # e.g. "MAYDAY", "PAN-PAN", "FUEL", "MED", None
    runway_needed: Optional[str]     # e.g. "27L", None if not requesting priority

    # ------------------------------------------------------------------
    # Derived convenience properties
    # ------------------------------------------------------------------

    @property
    def fuel_minutes_remaining(self) -> float:
        """Minutes of fuel above zero; does not account for reserve."""
        if self.fuel_burn_rate_kg_per_min <= 0:
            return float("inf")
        return self.fuel_kg / self.fuel_burn_rate_kg_per_min

    @property
    def fuel_minutes_above_reserve(self) -> float:
        """Minutes of usable fuel above the regulatory reserve."""
        usable = max(0.0, self.fuel_kg - self.reserve_fuel_kg)
        if self.fuel_burn_rate_kg_per_min <= 0:
            return float("inf")
        return usable / self.fuel_burn_rate_kg_per_min

    @property
    def is_fuel_critical(self) -> bool:
        """True when usable fuel above reserve drops below 30 minutes."""
        return self.fuel_minutes_above_reserve < 30.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AircraftState":
        return cls(
            id=d["id"],
            lat=d["lat"],
            lon=d["lon"],
            altitude_ft=d["altitude_ft"],
            heading_deg=d["heading_deg"],
            ground_speed_kt=d["ground_speed_kt"],
            fuel_kg=d["fuel_kg"],
            fuel_burn_rate_kg_per_min=d["fuel_burn_rate_kg_per_min"],
            reserve_fuel_kg=d["reserve_fuel_kg"],
            emergency_flag=d["emergency_flag"],
            emergency_type=d.get("emergency_type"),
            runway_needed=d.get("runway_needed"),
        )


@dataclass
class SectorState:
    aircraft: list[AircraftState]
    runway_availability: dict[str, bool]  # runway_id -> available
    sim_time_s: float                     # simulation elapsed time, seconds

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "aircraft": [a.to_dict() for a in self.aircraft],
            "runway_availability": self.runway_availability,
            "sim_time_s": self.sim_time_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SectorState":
        return cls(
            aircraft=[AircraftState.from_dict(a) for a in d["aircraft"]],
            runway_availability=d["runway_availability"],
            sim_time_s=d["sim_time_s"],
        )


# ---------------------------------------------------------------------------
# Mock factory — teammates can import this while BlueSky wiring is in progress
# ---------------------------------------------------------------------------

_AIRCRAFT_TYPES = ["A320", "B738", "A321", "B77W", "A333", "E190"]
_CALLSIGN_PREFIXES = ["BAW", "KLM", "DLH", "AFR", "UAE", "SWR", "IBE", "TAP"]
_EMERGENCY_TYPES = ["MAYDAY", "PAN-PAN", "FUEL", "MED", None, None, None, None]
_RUNWAYS = ["27L", "27R", "09L", "09R"]


def make_mock_sector(n: int = 12, seed: int = 42) -> SectorState:
    """
    Return a plausible SectorState for n aircraft over a generic TMA.

    The sector is loosely modeled on Amsterdam FIR airspace (52°N, 5°E).
    Two of the n aircraft will have active emergencies; one will be fuel-critical.
    """
    rng = random.Random(seed)

    used_callsigns: set[str] = set()

    def make_callsign() -> str:
        while True:
            cs = rng.choice(_CALLSIGN_PREFIXES) + str(rng.randint(100, 999))
            if cs not in used_callsigns:
                used_callsigns.add(cs)
                return cs

    aircraft: list[AircraftState] = []

    for i in range(n):
        is_emergency = i < 2
        is_fuel_critical = i == 0

        # Normal cruise parameters
        alt_ft = rng.choice([28000, 30000, 32000, 34000, 36000, 38000])
        burn_rate = rng.uniform(8.0, 14.0)      # kg/min, typical narrowbody
        reserve = rng.uniform(1800, 2400)        # kg

        if is_fuel_critical:
            # About 18 minutes above reserve — genuinely critical
            fuel_kg = reserve + burn_rate * rng.uniform(12, 20)
            emergency_type = "FUEL"
            emergency_flag = True
            runway_needed = rng.choice(_RUNWAYS)
        elif is_emergency:
            fuel_kg = reserve + burn_rate * rng.uniform(45, 90)
            emergency_type = rng.choice(["MAYDAY", "PAN-PAN", "MED"])
            emergency_flag = True
            runway_needed = rng.choice(_RUNWAYS)
        else:
            fuel_kg = reserve + burn_rate * rng.uniform(60, 240)
            emergency_type = None
            emergency_flag = False
            runway_needed = None

        aircraft.append(AircraftState(
            id=make_callsign(),
            lat=rng.uniform(51.0, 53.5),
            lon=rng.uniform(3.0, 7.5),
            altitude_ft=float(alt_ft),
            heading_deg=rng.uniform(0, 360),
            ground_speed_kt=rng.uniform(380, 480),
            fuel_kg=round(fuel_kg, 1),
            fuel_burn_rate_kg_per_min=round(burn_rate, 2),
            reserve_fuel_kg=round(reserve, 1),
            emergency_flag=emergency_flag,
            emergency_type=emergency_type,
            runway_needed=runway_needed,
        ))

    runway_availability = {rwy: rng.random() > 0.15 for rwy in _RUNWAYS}
    # Guarantee at least one runway is open
    if not any(runway_availability.values()):
        runway_availability[_RUNWAYS[0]] = True

    return SectorState(
        aircraft=aircraft,
        runway_availability=runway_availability,
        sim_time_s=rng.uniform(0, 3600),
    )


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sector = make_mock_sector(n=12)
    raw = sector.to_dict()
    restored = SectorState.from_dict(raw)

    assert len(restored.aircraft) == 12
    for orig, back in zip(sector.aircraft, restored.aircraft):
        assert orig.to_dict() == back.to_dict()

    print(json.dumps(raw, indent=2))
    print(f"\n{len(sector.aircraft)} aircraft, "
          f"{sum(a.emergency_flag for a in sector.aircraft)} emergencies, "
          f"runways: {sector.runway_availability}")
    print("Round-trip OK.")
