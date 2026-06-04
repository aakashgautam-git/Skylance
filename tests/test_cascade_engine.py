"""
Tests for cascade_engine.py.

Covers:
  - ICAO 5 nm / 1000 ft separation geometry (_check_separation)
  - Fuel exhaustion detection (CascadeEngine.evaluate)
  - All four action types (_apply_action)
  - CascadeEngine.evaluate and rank integration
"""

import pytest
from state_schema import AircraftState, SectorState, make_mock_sector
from cascade_engine import (
    CascadeEngine,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
    FuelBreach, SeparationBreach,
    _check_separation, _apply_action,
    _NM_TO_M, _FT_TO_M,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ac(id, lat, lon, *, alt_ft=30000, hdg=90.0, gs=450.0,
        fuel_kg=5000.0, burn=10.0, reserve=2000.0,
        emg=False, emg_type=None, rwy=None):
    return AircraftState(
        id=id, lat=lat, lon=lon,
        altitude_ft=float(alt_ft), heading_deg=float(hdg),
        ground_speed_kt=float(gs),
        fuel_kg=float(fuel_kg), fuel_burn_rate_kg_per_min=float(burn),
        reserve_fuel_kg=float(reserve),
        emergency_flag=emg, emergency_type=emg_type, runway_needed=rwy,
    )


def _sector(*aircraft, runways=None):
    rwa = runways if runways is not None else {"27L": True, "27R": True}
    return SectorState(aircraft=list(aircraft), runway_availability=rwa, sim_time_s=0.0)


@pytest.fixture
def engine():
    return CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)


# ---------------------------------------------------------------------------
# Separation geometry  (_check_separation, no engine)
# ---------------------------------------------------------------------------

class TestSeparationGeometry:

    def test_close_horizontal_and_close_vertical_is_breach(self):
        # 0.01° lon at 52°N ≈ 685 m ≈ 0.37 nm < 5 nm; 200 ft vertical < 1000 ft
        a1 = _ac("A1", 52.0, 4.00, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, alt_ft=30200)
        breaches = _check_separation(_sector(a1, a2))
        assert len(breaches) == 1
        b = breaches[0]
        assert frozenset([b.aircraft_id_1, b.aircraft_id_2]) == {"A1", "A2"}
        assert b.horizontal_nm < 5.0
        assert b.vertical_ft < 1000.0

    def test_close_horizontal_but_vertical_ok_no_breach(self):
        # 0.01° lon ≈ 0.37 nm, but 1001 ft vertical clears the 1000 ft minimum
        a1 = _ac("A1", 52.0, 4.00, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, alt_ft=31001)
        assert _check_separation(_sector(a1, a2)) == []

    def test_close_vertical_but_horizontal_ok_no_breach(self):
        # 0.1° lat ≈ 6 nm > 5 nm; 200 ft vertical
        a1 = _ac("A1", 52.0, 4.0, alt_ft=30000)
        a2 = _ac("A2", 52.1, 4.0, alt_ft=30200)
        assert _check_separation(_sector(a1, a2)) == []

    def test_exactly_5nm_horizontal_no_breach(self):
        # 5/60 degrees of latitude ≈ 5.003 nm via haversine > 5 nm threshold
        a1 = _ac("A1", 52.0, 4.0, alt_ft=30000)
        a2 = _ac("A2", 52.0 + 5.0 / 60, 4.0, alt_ft=30000)
        assert _check_separation(_sector(a1, a2)) == []

    def test_same_position_is_breach(self):
        a1 = _ac("A1", 52.0, 4.0, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.0, alt_ft=30000)
        assert len(_check_separation(_sector(a1, a2))) == 1

    def test_three_aircraft_detects_correct_pairs(self):
        # A1 and A2 are close; A3 is far away — only one pair should breach
        a1 = _ac("A1", 52.0, 4.00, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, alt_ft=30000)   # 0.37 nm from A1
        a3 = _ac("A3", 55.0, 4.00, alt_ft=30000)   # ~180 nm from both
        breaches = _check_separation(_sector(a1, a2, a3))
        assert len(breaches) == 1
        ids = frozenset([breaches[0].aircraft_id_1, breaches[0].aircraft_id_2])
        assert ids == {"A1", "A2"}

    def test_breach_fields_populated(self):
        a1 = _ac("A1", 52.0, 4.00, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, alt_ft=30200)
        b = _check_separation(_sector(a1, a2))[0]
        assert b.horizontal_nm > 0.0
        assert b.vertical_ft == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# Fuel exhaustion  (CascadeEngine.evaluate)
# ---------------------------------------------------------------------------

class TestFuelExhaustion:

    def test_fuel_breach_detected_within_horizon(self, engine):
        # 9.5 min above reserve → breach at the t=600 s checkpoint
        burn = 10.0; reserve = 2000.0
        a = _ac("LOW", 52.0, 4.0, fuel_kg=reserve + 9.5 * burn, burn=burn, reserve=reserve)
        score = engine.evaluate(_sector(a), NoAction())
        assert score.fuel_breach_count == 1
        assert score.fuel_breaches[0].aircraft_id == "LOW"

    def test_fuel_breach_elapsed_time_correct(self, engine):
        # 1 min + 0.5 kg above reserve → still safe at 60 s, breach at 120 s
        burn = 10.0; reserve = 2000.0
        fuel = reserve + burn + 0.5      # after 120 s: reserve + 0.5 - burn < reserve
        a = _ac("A", 52.0, 4.0, fuel_kg=fuel, burn=burn, reserve=reserve)
        score = engine.evaluate(_sector(a), NoAction())
        assert score.fuel_breach_count == 1
        assert score.fuel_breaches[0].elapsed_s == pytest.approx(120.0)

    def test_ample_fuel_no_breach(self, engine):
        burn = 10.0; reserve = 2000.0
        a = _ac("OK", 52.0, 4.0, fuel_kg=reserve + 200.0 * burn, burn=burn, reserve=reserve)
        assert engine.evaluate(_sector(a), NoAction()).fuel_breach_count == 0

    def test_first_breach_metadata_populated(self, engine):
        burn = 10.0; reserve = 2000.0
        a = _ac("LOW", 52.0, 4.0, fuel_kg=reserve + 5.0 * burn, burn=burn, reserve=reserve)
        score = engine.evaluate(_sector(a), NoAction())
        assert score.first_breach_type == "FUEL"
        assert score.first_breach_aircraft == "LOW"
        assert score.time_to_first_breach_s is not None

    def test_multiple_aircraft_each_fuel_breach_counted_once(self, engine):
        burn = 10.0; reserve = 2000.0
        a1 = _ac("A1", 52.0, 4.0, fuel_kg=reserve + 5.0 * burn, burn=burn, reserve=reserve)
        a2 = _ac("A2", 52.5, 4.5, fuel_kg=reserve + 7.0 * burn, burn=burn, reserve=reserve)
        score = engine.evaluate(_sector(a1, a2), NoAction())
        assert score.fuel_breach_count == 2
        ids = {b.aircraft_id for b in score.fuel_breaches}
        assert ids == {"A1", "A2"}


# ---------------------------------------------------------------------------
# Action types  (_apply_action — pure, no engine)
# ---------------------------------------------------------------------------

class TestActionTypes:

    def test_no_action_leaves_sector_unchanged(self):
        a = _ac("A1", 52.0, 4.0, hdg=90.0, gs=450.0)
        s = _sector(a)
        out = _apply_action(s, NoAction())
        ac = out.aircraft[0]
        assert ac.heading_deg == 90.0
        assert ac.ground_speed_kt == 450.0
        assert ac.lat == 52.0

    def test_assign_runway_sets_runway_needed(self):
        a = _ac("A1", 52.0, 4.0, rwy=None)
        out = _apply_action(_sector(a), AssignRunwayAction("A1", "27L"))
        assert out.aircraft[0].runway_needed == "27L"

    def test_assign_runway_does_not_move_aircraft(self):
        a = _ac("A1", 52.0, 4.0)
        out = _apply_action(_sector(a), AssignRunwayAction("A1", "27L"))
        assert out.aircraft[0].lat == 52.0
        assert out.aircraft[0].heading_deg == a.heading_deg

    def test_hold_reduces_speed_to_hold_speed(self):
        a = _ac("A1", 52.0, 4.0, gs=450.0)
        out = _apply_action(_sector(a), HoldAction("A1", hold_speed_kt=210.0))
        assert out.aircraft[0].ground_speed_kt == pytest.approx(210.0)

    def test_hold_does_not_increase_speed(self):
        # Aircraft already slower than hold speed: speed stays unchanged
        a = _ac("A1", 52.0, 4.0, gs=180.0)
        out = _apply_action(_sector(a), HoldAction("A1", hold_speed_kt=210.0))
        assert out.aircraft[0].ground_speed_kt == pytest.approx(180.0)

    def test_vector_changes_heading(self):
        a = _ac("A1", 52.0, 4.0, hdg=90.0)
        out = _apply_action(_sector(a), VectorAction("A1", new_heading_deg=180.0))
        assert out.aircraft[0].heading_deg == pytest.approx(180.0)

    def test_vector_wraps_heading_modulo_360(self):
        a = _ac("A1", 52.0, 4.0, hdg=350.0)
        out = _apply_action(_sector(a), VectorAction("A1", new_heading_deg=380.0))
        assert out.aircraft[0].heading_deg == pytest.approx(20.0)

    def test_action_only_changes_targeted_aircraft(self):
        a1 = _ac("A1", 52.0, 4.0, hdg=90.0)
        a2 = _ac("A2", 52.5, 4.5, hdg=270.0)
        out = _apply_action(_sector(a1, a2), VectorAction("A1", 0.0))
        assert out.aircraft[0].heading_deg == pytest.approx(0.0)   # changed
        assert out.aircraft[1].heading_deg == pytest.approx(270.0) # unchanged

    def test_apply_action_does_not_mutate_input(self):
        a = _ac("A1", 52.0, 4.0, hdg=90.0)
        s = _sector(a)
        _apply_action(s, VectorAction("A1", 0.0))
        assert s.aircraft[0].heading_deg == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# CascadeEngine integration
# ---------------------------------------------------------------------------

class TestCascadeEngine:

    def test_evaluate_safe_sector_no_breaches(self, engine):
        # Four aircraft at corners of a large box — well over 100 nm apart
        sector = _sector(
            _ac("A1", 51.0, 3.0, fuel_kg=8000, burn=10, reserve=2000),
            _ac("A2", 54.0, 3.0, fuel_kg=8000, burn=10, reserve=2000),
            _ac("A3", 51.0, 7.0, fuel_kg=8000, burn=10, reserve=2000),
            _ac("A4", 54.0, 7.0, fuel_kg=8000, burn=10, reserve=2000),
        )
        score = engine.evaluate(sector, NoAction())
        assert score.is_safe
        assert score.total_breach_count == 0

    def test_evaluate_detects_separation_breach(self, engine):
        # Two aircraft 0.37 nm apart, same heading and speed → stay close at every checkpoint
        a1 = _ac("A1", 52.0, 4.00, hdg=90, gs=450, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, hdg=90, gs=450, alt_ft=30200)
        score = engine.evaluate(_sector(a1, a2), NoAction())
        assert score.separation_breach_count >= 1
        assert score.first_breach_type == "SEPARATION"
        assert score.time_to_first_breach_s == pytest.approx(60.0)

    def test_vector_resolves_separation_breach(self, engine):
        # Same close pair; vectoring A2 north causes them to diverge
        a1 = _ac("A1", 52.0, 4.00, hdg=90, gs=450, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, hdg=90, gs=450, alt_ft=30200)
        sector = _sector(a1, a2)
        assert engine.evaluate(sector, NoAction()).separation_breach_count >= 1
        # After vector, A2 heads north while A1 heads east → >10 nm apart at t=60s
        score_vector = engine.evaluate(sector, VectorAction("A2", 0.0))
        assert score_vector.separation_breach_count == 0

    def test_hold_applies_reduced_speed_to_dead_reckoning(self, engine):
        # Single aircraft held at 210 kt: evaluate should not raise, still safe
        a = _ac("A1", 52.0, 4.0, gs=450, fuel_kg=8000, burn=10, reserve=2000)
        score = engine.evaluate(_sector(a), HoldAction("A1", 210.0))
        assert score.is_safe

    def test_assign_runway_does_not_affect_score_of_safe_sector(self, engine):
        # AssignRunway changes runway_needed but not position/speed → no new breaches
        a = _ac("A1", 52.0, 4.0, fuel_kg=8000, burn=10, reserve=2000,
                emg=True, emg_type="FUEL", rwy=None)
        sector = _sector(a, runways={"27L": True})
        base   = engine.evaluate(sector, NoAction())
        assign = engine.evaluate(sector, AssignRunwayAction("A1", "27L"))
        assert assign.fuel_breach_count == base.fuel_breach_count
        assert assign.separation_breach_count == base.separation_breach_count

    def test_rank_safe_action_before_unsafe(self, engine):
        # Vector resolves separation; NoAction doesn't → vector ranks first
        a1 = _ac("A1", 52.0, 4.00, hdg=90, gs=450, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, hdg=90, gs=450, alt_ft=30200)
        sector = _sector(a1, a2)
        ranked = engine.rank(sector, [NoAction(), VectorAction("A2", 0.0)])
        assert ranked[0][1].total_breach_count <= ranked[1][1].total_breach_count
        assert isinstance(ranked[0][0], VectorAction)

    def test_rank_breaks_ties_by_time_to_first_breach(self, engine):
        # Two breach-causing actions: one breaches later → ranks higher
        burn = 10.0; reserve = 2000.0
        a_early = _ac("EARLY", 52.0, 4.0, fuel_kg=reserve + 2*burn, burn=burn, reserve=reserve)
        a_late  = _ac("LATE",  52.5, 4.5, fuel_kg=reserve + 8*burn, burn=burn, reserve=reserve)
        sector = _sector(a_early, a_late)
        # HoldAction on EARLY vs NoAction: both have breaches, but hold doesn't help fuel
        no_score   = engine.evaluate(sector, NoAction())
        hold_score = engine.evaluate(sector, HoldAction("EARLY", 210.0))
        # Just verify rank is consistent with (total_breach_count, time_to_first_breach)
        ranked = engine.rank(sector, [NoAction(), HoldAction("EARLY", 210.0)])
        b0, b1 = ranked[0][1], ranked[1][1]
        if b0.total_breach_count == b1.total_breach_count:
            t0 = b0.time_to_first_breach_s or float("inf")
            t1 = b1.time_to_first_breach_s or float("inf")
            assert t0 >= t1

    def test_score_properties_consistent(self, engine):
        burn = 10.0; reserve = 2000.0
        a = _ac("A", 52.0, 4.0, fuel_kg=reserve + 5*burn, burn=burn, reserve=reserve)
        score = engine.evaluate(_sector(a), NoAction())
        assert score.total_breach_count == score.fuel_breach_count + score.separation_breach_count
        assert score.is_safe == (score.total_breach_count == 0)
        assert score.horizon_s == pytest.approx(600.0)
        assert score.checkpoint_interval_s == pytest.approx(60.0)
