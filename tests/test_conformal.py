"""
Tests for conformal.py.

Covers:
  - nc_score: fuel-breach regression, urgency ordering, symmetry
  - ConformalCertifier: calibration, certificate fields, error cases
  - Coverage: mathematical bound holds; fuel breaches captured after fix
"""

import matplotlib
matplotlib.use("Agg")   # headless — must precede any pyplot import

import pytest
from state_schema import AircraftState, SectorState, make_mock_sector
from cascade_engine import (
    CascadeEngine, CascadeScore,
    FuelBreach, SeparationBreach,
    NoAction,
)
from conformal import nc_score, ConformalCertifier, CalibrationPoint, check_coverage
from recommender import Recommender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_score(fuel=0, sep=0, first_elapsed_s=None, first_type=None,
                horizon=600.0):
    """Construct a minimal CascadeScore without running the engine."""
    fb = ([FuelBreach("T", 0.0, first_elapsed_s, 0.0, 100.0, 100.0)]
          if fuel > 0 else [])
    sb = ([SeparationBreach("T", "U", 0.0, first_elapsed_s, 3.0, 800.0)]
          if sep > 0 else [])
    return CascadeScore(
        action=NoAction(),
        horizon_s=horizon,
        checkpoint_interval_s=60.0,
        fuel_breach_count=fuel,
        separation_breach_count=sep,
        fuel_breaches=fb,
        separation_breaches=sb,
        time_to_first_breach_s=first_elapsed_s,
        first_breach_aircraft="T" if (fuel or sep) else None,
        first_breach_type=first_type,
    )


def _cal_points(n=60, seed=0):
    """Synthetic CalibrationPoints with a spread of nc values."""
    import random
    rng = random.Random(seed)
    pts = []
    for i in range(n):
        nc = float(i % 5) * 0.3           # nc in {0.0, 0.3, 0.6, 0.9, 1.2}
        held = (nc == 0.0) or rng.random() > 0.15
        pts.append(CalibrationPoint(i, 6, "mock", nc=nc, actually_held=held))
    return pts


# ---------------------------------------------------------------------------
# nc_score
# ---------------------------------------------------------------------------

class TestNcScore:

    def test_clean_scores_zero(self):
        assert nc_score(_mock_score(0, 0, None, None)) == 0.0

    def test_fuel_only_breach_scores_positive(self):
        # Regression: before the fix this returned 0.0
        assert nc_score(_mock_score(1, 0, 60.0, "FUEL")) > 0.0

    def test_sep_only_breach_scores_positive(self):
        assert nc_score(_mock_score(0, 1, 60.0, "SEPARATION")) > 0.0

    def test_fuel_and_sep_same_timing_score_equally(self):
        # After the fix, both breach types carry equal weight
        nc_f = nc_score(_mock_score(1, 0, 60.0, "FUEL"))
        nc_s = nc_score(_mock_score(0, 1, 60.0, "SEPARATION"))
        assert nc_f == nc_s

    def test_earlier_breach_scores_higher_than_later(self):
        nc_early = nc_score(_mock_score(1, 0,  60.0, "FUEL"))
        nc_late  = nc_score(_mock_score(1, 0, 540.0, "FUEL"))
        assert nc_early > nc_late

    def test_more_breaches_score_higher(self):
        nc_one = nc_score(_mock_score(1, 0, 60.0, "FUEL"))
        nc_two = nc_score(_mock_score(2, 0, 60.0, "FUEL"))
        assert nc_two > nc_one

    def test_nc_at_least_total_breach_count(self):
        # urgency ∈ [0, 1) so nc ≥ total_breach_count
        score = _mock_score(2, 1, 60.0, "SEPARATION")
        assert nc_score(score) >= 3

    def test_breach_at_very_end_of_horizon(self):
        # urgency ≈ 0 when breach is at t = horizon
        nc = nc_score(_mock_score(1, 0, 600.0, "FUEL", horizon=600.0))
        assert nc == pytest.approx(1.0, abs=0.01)

    def test_no_urgency_when_no_time_set(self):
        # time_to_first_breach_s = None → urgency = 0 → nc = total
        score = _mock_score(2, 0, None, None)
        # total_breach_count = 2, urgency = 0
        assert nc_score(score) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ConformalCertifier
# ---------------------------------------------------------------------------

class TestConformalCertifier:

    def test_raises_before_calibrate(self):
        cert = ConformalCertifier(alpha=0.05)
        with pytest.raises(RuntimeError):
            cert.certify(_mock_score(0, 0, None, None))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            ConformalCertifier(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalCertifier(alpha=1.0)
        with pytest.raises(ValueError):
            ConformalCertifier(alpha=1.5)

    def test_empty_calibration_raises(self):
        cert = ConformalCertifier(alpha=0.05)
        with pytest.raises(ValueError):
            cert.calibrate([])

    def test_clean_action_certified_safe(self):
        cert = ConformalCertifier(alpha=0.05)
        cert.calibrate(_cal_points())
        result = cert.certify(_mock_score(0, 0, None, None))
        assert result.certified_safe is True
        assert result.nc == pytest.approx(0.0)

    def test_high_nc_not_certified_safe(self):
        cert = ConformalCertifier(alpha=0.05)
        cert.calibrate(_cal_points())
        # nc = 10 + urgency >> any calibration threshold
        result = cert.certify(_mock_score(5, 5, 60.0, "FUEL"))
        assert result.certified_safe is False

    def test_certificate_fields_populated(self):
        cert = ConformalCertifier(alpha=0.05)
        cert.calibrate(_cal_points(n=60))
        result = cert.certify(_mock_score(0, 0, None, None))
        assert 0.0 <= result.p_value <= 1.0
        assert result.threshold >= 0.0
        assert result.n_calibration == 60
        assert result.confidence_pct == pytest.approx(95.0)
        assert result.alpha == pytest.approx(0.05)

    def test_threshold_increases_with_nc_quantile(self):
        # All-zero calibration → threshold = 0; adding high-nc points raises it
        pts_zero = [CalibrationPoint(i, 4, "t", nc=0.0, actually_held=True)
                    for i in range(40)]
        pts_high = [CalibrationPoint(i, 4, "t", nc=2.0, actually_held=False)
                    for i in range(40)]

        c_low = ConformalCertifier(alpha=0.05)
        c_low.calibrate(pts_zero)

        c_high = ConformalCertifier(alpha=0.05)
        c_high.calibrate(pts_high)

        assert c_high.threshold >= c_low.threshold

    def test_threshold_quantile_formula(self):
        # Exact formula check: n=19, alpha=0.10
        # q_level = ceil(20 * 0.90) / 19 = ceil(18.0) / 19 = 18/19 ≈ 0.947
        # The 94.7th percentile of {1,2,...,19} = the 18th value = 18.0
        alpha = 0.10
        pts = [CalibrationPoint(i, 4, "t", nc=float(i + 1), actually_held=True)
               for i in range(19)]
        cert = ConformalCertifier(alpha=alpha)
        cert.calibrate(pts)
        assert cert.threshold == pytest.approx(18.0)

    def test_fuel_breach_nc_exceeds_clean_nc(self):
        cert = ConformalCertifier(alpha=0.05)
        cert.calibrate(_cal_points())
        nc_clean = cert.certify(_mock_score(0, 0, None, None)).nc
        nc_fuel  = cert.certify(_mock_score(1, 0, 60.0, "FUEL")).nc
        assert nc_fuel > nc_clean


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestCoverage:

    def test_fuel_breach_scenarios_have_positive_nc(self):
        """
        Regression: nc_score must be > 0 for every fuel-breach scenario.
        Before the fix, fuel-only breaches returned nc = 0.
        """
        engine = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)
        burn = 10.0; reserve = 2000.0
        n_fuel_breach = 0

        for i in range(30):
            fuel = reserve + (3.0 + i % 7) * burn  # 3–9 min above reserve
            ac = AircraftState(
                id="LOW", lat=52.0, lon=float(4 + i * 0.05),
                altitude_ft=30000, heading_deg=90, ground_speed_kt=450,
                fuel_kg=fuel, fuel_burn_rate_kg_per_min=burn,
                reserve_fuel_kg=reserve,
                emergency_flag=True, emergency_type="FUEL", runway_needed="27L",
            )
            sector = SectorState([ac], {"27L": True}, sim_time_s=0.0)
            score = engine.evaluate(sector, NoAction())
            if score.fuel_breach_count > 0:
                n_fuel_breach += 1
                assert nc_score(score) > 0.0, (
                    f"Fuel breach (i={i}) scored nc=0 — nc_score fix not applied")

        assert n_fuel_breach > 0, "Test setup error: no fuel breaches observed"

    def test_coverage_bound_holds(self):
        """
        End-to-end coverage: empirical miscoverage ≤ alpha + 1 / (n_cal + 1).

        Uses CascadeEngine as both the nc_score predictor and the ground-truth
        oracle (is_safe), so there is no model mismatch and the conformal bound
        must hold deterministically on the held-out test split.
        """
        engine = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)
        alpha = 0.10
        points = []

        for seed in range(100):
            sector = make_mock_sector(n=6, seed=seed + 1000)
            score = engine.evaluate(sector, NoAction())
            points.append(CalibrationPoint(
                scenario_seed=seed, n_aircraft=6,
                action_type="NoAction",
                nc=nc_score(score),
                actually_held=score.is_safe,
            ))

        stats = check_coverage(points, alphas=[alpha], cal_fraction=0.8, show=False)
        n_cal = stats["n_cal"]
        tolerance = alpha + 1.0 / (n_cal + 1)
        empirical_miscov = stats["empirical_miscoverage"][0]
        assert empirical_miscov <= tolerance, (
            f"Coverage violated: miscoverage={empirical_miscov:.4f} "
            f"> alpha+1/(n+1)={tolerance:.4f}"
        )

    def test_coverage_bound_holds_with_fuel_breaches(self):
        """
        Same coverage bound specifically on scenarios where the cascade engine
        predicts a fuel breach.  This tests that the nc_score fix makes the
        guarantee meaningful for fuel safety — before the fix, all fuel-breach
        scenarios had nc=0 and would have been falsely certified safe.
        """
        engine = CascadeEngine(horizon_s=600.0, checkpoint_interval_s=60.0)
        alpha = 0.10
        burn = 10.0; reserve = 2000.0
        points = []

        for i in range(80):
            # Vary fuel level so nc values differ; all will breach within 600 s
            fuel = reserve + (1.0 + i % 9) * burn   # 1–9 min above reserve
            ac = AircraftState(
                id="LOW", lat=52.0, lon=float(4 + (i % 20) * 0.15),
                altitude_ft=30000, heading_deg=float(90 + i * 3),
                ground_speed_kt=450,
                fuel_kg=fuel, fuel_burn_rate_kg_per_min=burn,
                reserve_fuel_kg=reserve,
                emergency_flag=True, emergency_type="FUEL", runway_needed="27L",
            )
            sector = SectorState([ac], {"27L": True}, sim_time_s=0.0)
            score = engine.evaluate(sector, NoAction())

            # Every scenario here has a fuel breach
            assert score.fuel_breach_count > 0, \
                f"Expected fuel breach for i={i} (fuel={fuel:.1f})"
            nc = nc_score(score)
            # Regression: nc must be > 0 after the fix
            assert nc > 0.0, f"Fuel breach scenario i={i} scored nc=0"

            points.append(CalibrationPoint(
                scenario_seed=i, n_aircraft=1,
                action_type="NoAction",
                nc=nc,
                actually_held=False,   # all scenarios breach; ground truth = unsafe
            ))

        stats = check_coverage(points, alphas=[alpha], cal_fraction=0.8, show=False)
        n_cal = stats["n_cal"]
        tolerance = alpha + 1.0 / (n_cal + 1)
        empirical_miscov = stats["empirical_miscoverage"][0]
        assert empirical_miscov <= tolerance, (
            f"Fuel-breach coverage violated: {empirical_miscov:.4f} > {tolerance:.4f}"
        )
