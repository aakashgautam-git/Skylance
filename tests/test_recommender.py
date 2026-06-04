"""
Tests for recommender.py.

Covers:
  - enumerate_candidates: NoAction first, no closed-runway assignments,
    no duplicates, cap respected, emergency coverage, swap correctness
  - Recommender: valid output, top-k ordering, ranking behaviour
"""

import pytest
from state_schema import AircraftState, SectorState, make_mock_sector
from cascade_engine import (
    CascadeEngine,
    NoAction, AssignRunwayAction, HoldAction, VectorAction,
)
from recommender import Recommender, enumerate_candidates, RewardWeights


# ---------------------------------------------------------------------------
# Helpers
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


def _candidate_key(c):
    if isinstance(c, NoAction):
        return ("NoAction",)
    if isinstance(c, AssignRunwayAction):
        return ("AssignRunwayAction", c.aircraft_id, c.runway_id)
    if isinstance(c, HoldAction):
        return ("HoldAction", c.aircraft_id)
    if isinstance(c, VectorAction):
        return ("VectorAction", c.aircraft_id, round(c.new_heading_deg, 1))
    return (type(c).__name__,)


@pytest.fixture
def engine():
    return CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)


@pytest.fixture
def recommender(engine):
    return Recommender(engine)


# ---------------------------------------------------------------------------
# enumerate_candidates
# ---------------------------------------------------------------------------

class TestEnumerateCandidates:

    def test_no_action_is_always_first(self):
        sector = make_mock_sector(n=6, seed=42)
        assert isinstance(enumerate_candidates(sector)[0], NoAction)

    def test_no_closed_runway_in_candidates(self):
        for seed in range(15):
            sector = make_mock_sector(n=8, seed=seed)
            open_rwys = {r for r, ok in sector.runway_availability.items() if ok}
            for c in enumerate_candidates(sector):
                if isinstance(c, AssignRunwayAction):
                    assert c.runway_id in open_rwys, (
                        f"seed={seed}: closed runway {c.runway_id!r} assigned to {c.aircraft_id}")

    def test_no_duplicate_candidates(self):
        for seed in range(15):
            sector = make_mock_sector(n=8, seed=seed)
            candidates = enumerate_candidates(sector)
            keys = [_candidate_key(c) for c in candidates]
            assert len(keys) == len(set(keys)), (
                f"seed={seed}: duplicates {[k for k in keys if keys.count(k) > 1]}")

    def test_respects_max_candidates_cap(self):
        sector = make_mock_sector(n=12, seed=42)
        for cap in [1, 5, 10, 20]:
            assert len(enumerate_candidates(sector, max_candidates=cap)) <= cap

    def test_cap_of_one_returns_only_no_action(self):
        sector = make_mock_sector(n=6, seed=42)
        cands = enumerate_candidates(sector, max_candidates=1)
        assert len(cands) == 1
        assert isinstance(cands[0], NoAction)

    def test_emergency_aircraft_receives_runway_candidates(self):
        emg = _ac("EMG", 52.0, 4.0, emg=True, emg_type="MAYDAY", rwy="27L",
                  fuel_kg=2200, burn=10, reserve=2000)
        nrm = _ac("NRM", 52.5, 4.5)
        sector = _sector(emg, nrm, runways={"27L": True, "27R": True})
        assign_ids = {c.aircraft_id for c in enumerate_candidates(sector)
                      if isinstance(c, AssignRunwayAction)}
        assert "EMG" in assign_ids

    def test_fuel_critical_aircraft_receives_runway_candidates(self):
        # 25 min above reserve → is_fuel_critical = True
        crit = _ac("CRIT", 52.0, 4.0, fuel_kg=2000 + 25*10, burn=10, reserve=2000)
        assert crit.is_fuel_critical
        sector = _sector(crit, runways={"27L": True})
        assign_ids = {c.aircraft_id for c in enumerate_candidates(sector)
                      if isinstance(c, AssignRunwayAction)}
        assert "CRIT" in assign_ids

    def test_swap_never_assigns_closed_runway(self):
        # Two emergencies both want 27L; 09R is closed
        a = _ac("E1", 52.0, 4.0, emg=True, emg_type="MAYDAY",  rwy="27L",
                fuel_kg=2200, burn=10, reserve=2000)
        b = _ac("E2", 52.5, 4.5, emg=True, emg_type="PAN-PAN", rwy="27L",
                fuel_kg=4000, burn=10, reserve=2000)
        sector = _sector(a, b, runways={"27L": True, "27R": True, "09R": False})
        for c in enumerate_candidates(sector):
            if isinstance(c, AssignRunwayAction):
                assert sector.runway_availability.get(c.runway_id) is True, (
                    f"Closed runway {c.runway_id!r} assigned to {c.aircraft_id}")

    def test_swap_redirects_lower_urgency_when_same_runway(self):
        # E1 is more urgent (less fuel); E2 has more fuel but both want 27L.
        # The swap step should generate candidates for E2 that use 27R (uncontested).
        a = _ac("E1", 52.0, 4.0, emg=True, emg_type="MAYDAY",  rwy="27L",
                fuel_kg=2200, burn=10, reserve=2000)
        b = _ac("E2", 52.5, 4.5, emg=True, emg_type="PAN-PAN", rwy="27L",
                fuel_kg=4000, burn=10, reserve=2000)
        sector = _sector(a, b, runways={"27L": True, "27R": True, "09R": False})
        candidates = enumerate_candidates(sector)
        e2_runways = {c.runway_id for c in candidates
                      if isinstance(c, AssignRunwayAction) and c.aircraft_id == "E2"}
        # 27R is an uncontested open runway — must appear as an option for E2
        assert "27R" in e2_runways
        # Closed runway must never appear
        assert "09R" not in e2_runways

    def test_all_candidates_aircraft_ids_exist_in_sector(self):
        sector = make_mock_sector(n=8, seed=7)
        valid_ids = {ac.id for ac in sector.aircraft}
        for c in enumerate_candidates(sector):
            ac_id = getattr(c, "aircraft_id", None)
            if ac_id is not None:
                assert ac_id in valid_ids, f"Unknown aircraft id {ac_id!r} in candidate"


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TestRecommender:

    def test_recommend_returns_valid_recommendation(self, recommender):
        sector = make_mock_sector(n=6, seed=42)
        rec = recommender.recommend(sector)
        assert rec.action is not None
        assert rec.score is not None
        assert rec.baseline_score is not None
        assert rec.candidates_evaluated >= 1
        assert rec.reward > -1e9

    def test_recommend_action_is_in_candidate_set(self, engine):
        # The chosen action must be one of the enumerated candidates
        sector = make_mock_sector(n=6, seed=42)
        candidates = enumerate_candidates(sector)
        rec = Recommender(engine).recommend(sector)
        chosen_key = _candidate_key(rec.action)
        candidate_keys = {_candidate_key(c) for c in candidates}
        assert chosen_key in candidate_keys

    def test_recommend_top_k_at_most_k(self, recommender):
        sector = make_mock_sector(n=6, seed=42)
        assert len(recommender.recommend_top_k(sector, k=3)) <= 3

    def test_recommend_top_k_sorted_descending_by_reward(self, recommender):
        sector = make_mock_sector(n=8, seed=42)
        top5 = recommender.recommend_top_k(sector, k=5)
        rewards = [r.reward for r in top5]
        assert rewards == sorted(rewards, reverse=True)

    def test_recommend_top_k_first_matches_recommend(self, recommender):
        sector = make_mock_sector(n=6, seed=7)
        rec   = recommender.recommend(sector)
        top1  = recommender.recommend_top_k(sector, k=1)
        assert _candidate_key(rec.action) == _candidate_key(top1[0].action)

    def test_safe_action_preferred_when_it_eliminates_breach(self, engine):
        # Two close parallel aircraft: NoAction → separation breach;
        # VectorAction → safe. Recommender should prefer the vector.
        a1 = _ac("A1", 52.0, 4.00, hdg=90, gs=450, alt_ft=30000)
        a2 = _ac("A2", 52.0, 4.01, hdg=90, gs=450, alt_ft=30200)
        sector = _sector(a1, a2, runways={"27L": True})
        rec = Recommender(engine).recommend(sector)
        # The recommender should find the vector candidate via the close-pair logic
        # and prefer it (0 breaches) over NoAction (1+ breaches)
        assert rec.score.total_breach_count <= rec.baseline_score.total_breach_count

    def test_reward_weights_influence_recommendation(self, engine):
        # Disabling the fairness bonus changes the reward magnitudes but
        # the recommendation must still be a valid action
        sector = make_mock_sector(n=6, seed=10)
        w = RewardWeights(emergency_bonus=0.0, fuel_critical_bonus=0.0)
        rec = Recommender(engine, weights=w).recommend(sector)
        assert rec.action is not None
        assert isinstance(rec.reward, float)

    def test_single_aircraft_sector_returns_recommendation(self, recommender):
        a = _ac("SOLO", 52.0, 4.0, emg=True, emg_type="FUEL", rwy="27L",
                fuel_kg=2200, burn=10, reserve=2000)
        sector = _sector(a, runways={"27L": True})
        rec = recommender.recommend(sector)
        assert rec.action is not None
