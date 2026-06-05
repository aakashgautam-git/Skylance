"""
SKYLANCE-X — ATC Emergency Suggestion Engine Dashboard
Streamlit front-end.

All mutable state lives in st.session_state.  No global mutables.

Session-state keys
------------------
  initialized       bool          first-run flag
  adapter           BlueSkyAdapter
  engine            CascadeEngine
  recommender       Recommender   (SKYLANCE-X)
  baseline          RFBaseline
  certifier         ConformalCertifier
  explainer         Explainer
  sector            SectorState   ← single source of truth for displayed state
  sector_seed       int
  n_aircraft        int
  mode              "SKYLANCE-X" | "Baseline"
  sx_reco           Recommendation
  bl_action         CandidateAction
  bl_score          CascadeScore
  certificate       Certificate
  counterfactual    Counterfactual | None
  needs_recompute   bool
  baseline_plan     list[dict] | None  ← landing plan snapshot before emergency
  current_plan      list[dict] | None  ← landing plan snapshot after SKYLANCE-X replanning
  baseline_locked   bool               ← True once an emergency has been declared
"""

import math
import sys
import os

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bluesky"))
sys.path.insert(0, os.path.dirname(__file__))

# ── page config (must come before any other st call) ──────────────────────────
st.set_page_config(
    page_title="SKYLANCE-X",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ──────────────────────────────────────────────────────────────────
_BG       = "#0d1117"
_GRID     = "#1c2128"   # neutral dark — not green-tinted
_GREEN    = "#3fb950"
_ORANGE   = "#e3a135"
_RED      = "#f85149"
_BLUE     = "#58a6ff"
_TEXT     = "#c9d1d9"
_MUTED    = "#6e7681"

# Radar-scope palette (professional ATC look) ──────────────────────────────────
_RADAR_BG   = "#05101e"                  # deep navy paper / plot background
_RADAR_FACE = "rgba(16,34,58,0.55)"      # scope face — lighter than corners (vignette)
_RING_COL   = "rgba(99,160,220,0.16)"    # concentric range rings
_RADIAL_COL = "rgba(99,160,220,0.09)"    # radial bearing lines
_PLANE_NORM = "#d7dde6"                  # normal aircraft glyph (light grey/white)

_CSS = """
<style>
/* Hide the Streamlit toolbar/decoration for a clean look, but keep the header
   element itself so its sidebar expand control still works. */
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none; }
header[data-testid="stHeader"] { background: transparent; box-shadow: none; }
.block-container { padding-top: 1.2rem; }
/* Keep the sidebar (airport/runway controls) always visible — it must never
   collapse out of reach. */
section[data-testid="stSidebar"] {
    transform: none !important;
    visibility: visible !important;
    min-width: 244px !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] section { padding-top: 0.5rem; }
div[data-testid="stButton"] button {
    background: #161b22;
    border: 1px solid #30363d;
    color: #c9d1d9;
    font-size: 13px;
    font-weight: 400;
    border-radius: 5px;
    transition: border-color 0.15s, color 0.15s;
}
div[data-testid="stButton"] button:hover {
    border-color: #58a6ff;
    color: #e6edf3;
    background: #161b22;
}
div[data-testid="stRadio"] > div { gap: 6px; }
</style>
"""

# Airport reference points for the radar.  Runways are grouped here so
# both the radar markers and the sidebar status panel stay in sync.
_AIRPORTS: dict[str, dict] = {
    'VOBL': {'lat': 13.20, 'lon': 77.71, 'runways': ['09R', '27L'],
             'city': 'Bengaluru', 'country': 'IN'},
    'VABB': {'lat': 19.09, 'lon': 72.87, 'runways': ['09', '27'],
             'city': 'Mumbai', 'country': 'IN'},
    'VIDP': {'lat': 28.56, 'lon': 77.10, 'runways': ['10', '28'],
             'city': 'Delhi', 'country': 'IN'},
    'VOMM': {'lat': 12.99, 'lon': 80.17, 'runways': ['07', '25'],
             'city': 'Chennai', 'country': 'IN'},
    'VOHS': {'lat': 17.24, 'lon': 78.43, 'runways': ['09L', '27R'],
             'city': 'Hyderabad', 'country': 'IN'},
}
_RUNWAY_AIRPORT = {
    rwy: icao
    for icao, ap in _AIRPORTS.items()
    for rwy in ap['runways']
}


# ── landing-plan helpers ───────────────────────────────────────────────────────

def _nearest_airport_runway(sector, ac) -> tuple:
    """Return (icao, runway_id) for the nearest airport with an open runway."""
    best_icao, best_rwy, best_dist = None, None, float('inf')
    for icao, ap in _AIRPORTS.items():
        dist = math.sqrt((ap['lat'] - ac.lat) ** 2 + (ap['lon'] - ac.lon) ** 2)
        open_rwys = [r for r in ap['runways'] if sector.runway_availability.get(r, False)]
        if open_rwys and dist < best_dist:
            best_dist = dist
            best_icao = icao
            best_rwy  = open_rwys[0]
    return best_icao, best_rwy


def _bearing_to(lat: float, lon: float, tlat: float, tlon: float) -> float:
    """Local bearing (deg true, 0–360) from (lat,lon) toward a target point.

    Uses the same flat-earth convention as `_project_position`: heading 0 = north
    (lat increases), heading 90 = east (lon increases, scaled by cos lat).
    """
    de = (tlon - lon) * math.cos(math.radians(lat))   # east displacement, lat-deg equiv
    dn = (tlat - lat)                                  # north displacement
    return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0


def _advance_toward(lat: float, lon: float, gs_kt: float,
                    tlat: float, tlon: float, seconds: float) -> tuple:
    """
    Move straight toward a target threshold, clamped so it never overshoots.

    Returns (new_lat, new_lon, new_heading_deg, landed).  Distance to target
    (in the hypot·60 nm metric used for ETA everywhere) decreases by exactly the
    step length each call, so ETA is monotonically non-increasing while inbound.
    """
    dlat, dlon = tlat - lat, tlon - lon
    rem_nm  = math.hypot(dlat, dlon) * 60.0
    step_nm = gs_kt * (seconds / 3600.0)
    hdg     = _bearing_to(lat, lon, tlat, tlon)
    if rem_nm <= 1e-6 or step_nm >= rem_nm:
        return tlat, tlon, hdg, True
    frac = step_nm / rem_nm
    return lat + dlat * frac, lon + dlon * frac, hdg, False


def _compute_landing_plan(sector, action=None, assignments=None) -> list:
    """
    Derive a per-aircraft landing plan from sector state + optional action.

    `assignments` (callsign -> runway_id) is the persistent, locked set of runway
    assignments the aircraft are actually flying toward; it takes priority so the
    Before/After table always agrees with the rerouted path drawn on the map.

    Each entry: {callsign, assigned_airport, assigned_runway, eta_min, eta_nm,
                 sequence_position, is_emergency, emergency_type, is_action_target}
    """
    from cascade_engine import AssignRunwayAction

    assignments = assignments or {}
    action_ac_id = None
    if isinstance(action, AssignRunwayAction):
        action_ac_id = action.aircraft_id

    plans = []
    for ac in sector.aircraft:
        if ac.id in assignments:
            rwy = assignments[ac.id]
            ap  = _RUNWAY_AIRPORT.get(rwy)
        elif ac.runway_needed and sector.runway_availability.get(ac.runway_needed, False):
            rwy = ac.runway_needed
            ap  = _RUNWAY_AIRPORT.get(rwy)
        else:
            ap, rwy = _nearest_airport_runway(sector, ac)

        eta_min = eta_nm = None
        if ap and ap in _AIRPORTS:
            apd     = _AIRPORTS[ap]
            eta_nm  = math.sqrt((apd['lat'] - ac.lat) ** 2 + (apd['lon'] - ac.lon) ** 2) * 60
            eta_min = eta_nm / max(ac.ground_speed_kt, 1) * 60

        plans.append({
            "callsign":          ac.id,
            "assigned_airport":  ap,
            "assigned_runway":   rwy,
            "eta_min":           eta_min,
            "eta_nm":            eta_nm,
            "is_emergency":      ac.emergency_flag,
            "emergency_type":    ac.emergency_type,
            "is_action_target":  ac.id == action_ac_id or ac.id in assignments,
            "sequence_position": 0,   # filled below
        })

    plans.sort(key=lambda p: (0 if p["is_emergency"] else 1, p["eta_min"] or 999.0))
    for i, p in enumerate(plans):
        p["sequence_position"] = i + 1
    return plans


def _resolve_diversions(sector, assignments) -> dict:
    """
    Capacity-aware runway allocation around an emergency.

    When an emergency / priority approach claims a runway at airport X, the open
    runways at X are a finite resource.  Non-emergency aircraft inbound to X fill
    the remaining runways closest-first; any that no longer fit are DIVERTED to
    the nearest alternate airport that still has a free runway.

    Returns {callsign: (alt_airport_icao, alt_runway_id)} for diverted aircraft.
    """
    by_id     = {a.id: a for a in sector.aircraft}
    open_rwys = {icao: [r for r in ap['runways']
                        if sector.runway_availability.get(r, False)]
                 for icao, ap in _AIRPORTS.items()}
    used      = {icao: set() for icao in _AIRPORTS}

    # Reserve the runways already locked to assignments; note which airports an
    # emergency aircraft has taken (those are the ones that get contested).
    emergency_airports: set[str] = set()
    for cs, rwy in assignments.items():
        icao = _RUNWAY_AIRPORT.get(rwy)
        if not icao:
            continue
        used[icao].add(rwy)
        if by_id.get(cs) and by_id[cs].emergency_flag:
            emergency_airports.add(icao)

    if not emergency_airports:
        return {}

    def _dist(ac, icao):
        ap = _AIRPORTS[icao]
        return math.hypot(ap['lat'] - ac.lat, ap['lon'] - ac.lon)

    def _nearest_free(ac, exclude):
        best, best_d = None, float('inf')
        for icao in _AIRPORTS:
            if icao in exclude:
                continue
            free = [r for r in open_rwys[icao] if r not in used[icao]]
            if free and _dist(ac, icao) < best_d:
                best_d, best = _dist(ac, icao), (icao, free[0])
        return best

    # Non-emergency, unassigned aircraft whose nearest airport is an emergency
    # airport — they compete for whatever runways remain there.
    competitors = []
    for a in sector.aircraft:
        if a.emergency_flag or a.id in assignments:
            continue
        intended, _ = _nearest_airport_runway(sector, a)
        if intended in emergency_airports:
            competitors.append((a, intended))
    competitors.sort(key=lambda t: _dist(t[0], t[1]))   # closest keeps the runway

    diversions = {}
    for a, intended in competitors:
        free_here = [r for r in open_rwys[intended] if r not in used[intended]]
        if free_here:
            used[intended].add(free_here[0])        # still fits — lands at intended
            continue
        alt = _nearest_free(a, exclude={intended})   # runways full — divert away
        if alt:
            used[alt[0]].add(alt[1])
            diversions[a.id] = alt
    return diversions


def _diff_plans(baseline: list, current: list) -> list:
    """
    Per-callsign diff between baseline and current landing plans.

    Each entry: {callsign, old_runway, new_runway, old_airport, new_airport,
                 runway_changed, airport_changed, eta_delta_min, sequence_delta,
                 old_sequence, new_sequence, old_eta_min, new_eta_min,
                 status (unchanged|rerouted|resequenced|priority),
                 is_emergency, emergency_type}
    """
    b_map = {p["callsign"]: p for p in baseline}
    c_map = {p["callsign"]: p for p in current}
    diffs = []

    for cs in set(b_map) | set(c_map):
        b = b_map.get(cs)
        c = c_map.get(cs)
        if b is None or c is None:
            continue

        rwy_changed = b["assigned_runway"]  != c["assigned_runway"]
        ap_changed  = b["assigned_airport"] != c["assigned_airport"]
        eta_delta   = (
            round(c["eta_min"] - b["eta_min"], 1)
            if b["eta_min"] is not None and c["eta_min"] is not None else None
        )
        seq_delta = c["sequence_position"] - b["sequence_position"]

        if c.get("is_action_target") and c.get("is_emergency"):
            status = "priority"
        elif rwy_changed or ap_changed:
            status = "rerouted"
        elif seq_delta != 0:
            status = "resequenced"
        else:
            status = "unchanged"

        diffs.append({
            "callsign":       cs,
            "old_runway":     b["assigned_runway"],
            "new_runway":     c["assigned_runway"],
            "old_airport":    b["assigned_airport"],
            "new_airport":    c["assigned_airport"],
            "runway_changed": rwy_changed,
            "airport_changed":ap_changed,
            "eta_delta_min":  eta_delta,
            "sequence_delta": seq_delta,
            "old_sequence":   b["sequence_position"],
            "new_sequence":   c["sequence_position"],
            "old_eta_min":    b["eta_min"],
            "new_eta_min":    c["eta_min"],
            "status":         status,
            "is_emergency":   c.get("is_emergency", False),
            "emergency_type": c.get("emergency_type"),
        })

    diffs.sort(key=lambda d: (
        0 if d["status"] == "priority" else
        1 if d["status"] in ("rerouted", "resequenced") else 2,
        d["new_sequence"] or 99,
    ))
    return diffs


# ── radar helpers ──────────────────────────────────────────────────────────────

def _project_position(lat: float, lon: float, hdg_deg: float,
                      gs_kt: float, minutes: float) -> tuple[float, float]:
    """Dead-reckon a position forward by `minutes` of flight."""
    dist_nm = gs_kt * (minutes / 60.0)
    hdg_rad = math.radians(hdg_deg)
    lat_rad = math.radians(lat)
    new_lat = lat + dist_nm * math.cos(hdg_rad) / 60.0
    new_lon = lon + dist_nm * math.sin(hdg_rad) / (60.0 * max(math.cos(lat_rad), 1e-9))
    return new_lat, new_lon


# ══════════════════════════════════════════════════════════════════════════════
# Initialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_initialized():
    """
    Create all heavy objects on first visit and store them in st.session_state.

    Everything lives in session_state — no @st.cache_resource — so BlueSky's
    module-level singletons (bs.traf, bs.sim, …) are always owned by a single
    execution context and cannot be shared across Streamlit reruns or threads.
    """
    if st.session_state.get("initialized"):
        return

    with st.spinner("Initialising SKYLANCE-X — one-time setup (~30 s)…"):
        from bluesky_adapter import BlueSkyAdapter
        from cascade_engine  import CascadeEngine, _check_separation
        from recommender     import Recommender
        from baseline        import RFBaseline
        from explainer       import Explainer
        from conformal       import ConformalCertifier, CalibrationPoint
        from state_schema    import make_mock_sector

        adapter     = BlueSkyAdapter(simdt=1.0)   # 10× faster than default 0.05s
        engine      = CascadeEngine(adapter, horizon_s=300.0, checkpoint_interval_s=60.0)
        recommender = Recommender(engine, max_candidates=10)
        baseline    = RFBaseline(auto_train=True, n_train=200)
        explainer   = Explainer(recommender)

        # Fast geometric certifier calibration (no BlueSky simulation needed)
        certifier = ConformalCertifier(alpha=0.05)
        cal_pts: list[CalibrationPoint] = []
        for seed in range(40):
            sec = make_mock_sector(n=8, seed=seed + 700)
            seps = _check_separation(sec)
            nc = 1.0 if seps else 0.0
            cal_pts.append(CalibrationPoint(
                scenario_seed=seed, n_aircraft=8,
                action_type="GeomCheck",
                nc=nc, actually_held=not seps,
            ))
        certifier.calibrate(cal_pts)

        st.session_state.adapter     = adapter
        st.session_state.engine      = engine
        st.session_state.recommender = recommender
        st.session_state.baseline    = baseline
        st.session_state.certifier   = certifier
        st.session_state.explainer   = explainer
        st.session_state.mode            = "SKYLANCE-X"
        st.session_state.sector_seed     = 42
        st.session_state.n_aircraft      = 10
        initial_sector = make_mock_sector(n=10, seed=42)
        st.session_state.sector          = initial_sector
        st.session_state.baseline_plan   = _compute_landing_plan(initial_sector)
        st.session_state.current_plan    = None
        st.session_state.baseline_locked = False
        st.session_state.assignments     = {}   # callsign -> locked runway flown toward
        st.session_state.original_paths  = {}   # callsign -> frozen pre-emergency plan
        st.session_state.diversions      = {}   # callsign -> (alt_airport, alt_runway)
        st.session_state.needs_recompute = True
        st.session_state.initialized     = True


def _recompute():
    """
    Run recommender + certifier + baseline on the current sector.

    The Explainer is NOT called here — it can make 50+ recommender calls and
    would block the UI for ~60 s.  Run it on demand via the Explain button.
    """
    sector      = st.session_state.sector
    recommender = st.session_state.recommender
    baseline    = st.session_state.baseline
    certifier   = st.session_state.certifier
    engine      = st.session_state.engine

    # SKYLANCE-X (10 candidates × 5 checkpoints ≈ 7 s with simdt=1.0)
    sx_reco = recommender.recommend(sector)
    cert    = certifier.certify(sx_reco.score)

    # Baseline evaluated on same sector for a fair comparison
    bl_action = baseline.recommend(sector)
    bl_score  = engine.evaluate(sector, bl_action)

    # Lock the runway assignment the first time SKYLANCE-X assigns this aircraft,
    # so it keeps flying toward a FIXED threshold (ETA stays monotonic) and the
    # map + Before/After table never disagree on the runway.
    from cascade_engine import AssignRunwayAction
    assignments = st.session_state.setdefault("assignments", {})
    diversions  = st.session_state.setdefault("diversions", {})
    if isinstance(sx_reco.action, AssignRunwayAction):
        assignments.setdefault(sx_reco.action.aircraft_id, sx_reco.action.runway_id)

    # Capacity-aware diversion: when the emergency fills its airport's runways,
    # bump the non-emergency flights that no longer fit to the nearest alternate
    # airport.  Locking them into `assignments` makes them actually fly there
    # (great-circle propagation) and draw their new route on the map.
    for cs, (alt_icao, alt_rwy) in _resolve_diversions(sector, assignments).items():
        if cs not in assignments:
            assignments[cs] = alt_rwy
            diversions[cs]  = (alt_icao, alt_rwy)

    st.session_state.sx_reco          = sx_reco
    st.session_state.certificate      = cert
    st.session_state.counterfactual   = None   # cleared; request via Explain button
    st.session_state.bl_action        = bl_action
    st.session_state.bl_score         = bl_score
    st.session_state.current_plan     = _compute_landing_plan(
        sector, sx_reco.action, assignments)
    st.session_state.needs_recompute  = False


def _run_explainer():
    """Run the counterfactual explainer on demand (slow — ~30-60 s)."""
    sector  = st.session_state.sector
    sx_reco = st.session_state.sx_reco
    if sx_reco is None:
        return
    with st.status("Computing explanation…", expanded=True) as _status:
        st.write("Searching for the minimal single-feature change "
                 "that would flip the recommendation…")
        cf = st.session_state.explainer.explain(sector, sx_reco)
        if cf is not None:
            _status.update(label="Explanation ready", state="complete", expanded=False)
        else:
            _status.update(label="No single-feature flip found within realistic bounds",
                           state="complete", expanded=False)
    st.session_state.counterfactual = cf


# ══════════════════════════════════════════════════════════════════════════════
# Sector mutation helpers  (each marks needs_recompute)
# ══════════════════════════════════════════════════════════════════════════════

def _load_new_sector(seed: int, n: int):
    from state_schema import make_mock_sector
    new_sector = make_mock_sector(n=n, seed=seed)
    st.session_state.sector          = new_sector
    st.session_state.baseline_plan   = _compute_landing_plan(new_sector)
    st.session_state.current_plan    = None
    st.session_state.baseline_locked = False
    st.session_state.assignments     = {}
    st.session_state.original_paths  = {}
    st.session_state.diversions      = {}
    st.session_state.needs_recompute = True


def _step_60s():
    """
    Advance the sim 60 s with pure-Python dead-reckoning.

    BlueSky is intentionally NOT used here: its `bs.init` is incompatible with
    this environment (raises 'Traffic' object has no attribute 'groups'), and
    every other module — cascade engine, recommender — is already pure Python.
    Assigned aircraft fly straight at their runway threshold (clamped, so ETA
    decreases every step until touchdown); all others coast on their heading.
    """
    from state_schema import AircraftState, SectorState
    sector      = st.session_state.sector
    assignments = st.session_state.get("assignments", {})
    DT_MIN = 1.0   # 60 s

    new_acs = []
    for ac in sector.aircraft:
        rwy  = assignments.get(ac.id)
        icao = _RUNWAY_AIRPORT.get(rwy) if rwy else None
        ap   = _AIRPORTS.get(icao) if icao else None
        if ap:   # assigned → converge on the runway threshold
            nlat, nlon, nhdg, _landed = _advance_toward(
                ac.lat, ac.lon, ac.ground_speed_kt, ap['lat'], ap['lon'], 60.0)
        else:    # unassigned → coast along current heading
            nlat, nlon = _project_position(
                ac.lat, ac.lon, ac.heading_deg, ac.ground_speed_kt, DT_MIN)
            nhdg = ac.heading_deg
        new_fuel = max(0.0, ac.fuel_kg - ac.fuel_burn_rate_kg_per_min * DT_MIN)
        new_acs.append(AircraftState(
            id=ac.id, lat=nlat, lon=nlon, altitude_ft=ac.altitude_ft,
            heading_deg=nhdg, ground_speed_kt=ac.ground_speed_kt,
            fuel_kg=round(new_fuel, 1),
            fuel_burn_rate_kg_per_min=ac.fuel_burn_rate_kg_per_min,
            reserve_fuel_kg=ac.reserve_fuel_kg,
            emergency_flag=ac.emergency_flag, emergency_type=ac.emergency_type,
            runway_needed=ac.runway_needed, vertical_speed_fpm=ac.vertical_speed_fpm,
        ))

    new_sector = SectorState(
        aircraft=new_acs,
        runway_availability=dict(sector.runway_availability),
        sim_time_s=sector.sim_time_s + 60.0,
        wind_north_kt=sector.wind_north_kt,
        wind_east_kt=sector.wind_east_kt,
    )
    st.session_state.sector = new_sector
    if not st.session_state.get("baseline_locked", False):
        st.session_state.baseline_plan = _compute_landing_plan(new_sector)
    st.session_state.needs_recompute = True


def _inject_emergency(aircraft_id: str, emg_type: str):
    from state_schema import AircraftState, SectorState
    st.session_state.baseline_locked = True
    sector = st.session_state.sector

    # Freeze each aircraft's ORIGINAL planned path (BUG 2): its position at the
    # moment of declaration → its pre-emergency assigned runway threshold.  Drawn
    # later as the dashed line so the reroute divergence is obvious in a demo.
    original_paths = st.session_state.setdefault("original_paths", {})
    b_map = {p["callsign"]: p
             for p in (st.session_state.get("baseline_plan") or [])}
    for ac in sector.aircraft:
        if ac.id in original_paths:
            continue
        b = b_map.get(ac.id)
        if not b or not b.get("assigned_airport"):
            continue
        original_paths[ac.id] = {
            "from_lat": ac.lat, "from_lon": ac.lon,
            "airport":  b["assigned_airport"],
            "runway":   b["assigned_runway"],
        }

    def _patch(ac: AircraftState) -> AircraftState:
        if ac.id != aircraft_id:
            return ac
        # Default the requested runway to the NEAREST open one (not the first in
        # the dict) so an emergency heads to the closest airport, not Bengaluru.
        near_rwy = _nearest_airport_runway(sector, ac)[1]
        return AircraftState(
            id=ac.id, lat=ac.lat, lon=ac.lon,
            altitude_ft=ac.altitude_ft, heading_deg=ac.heading_deg,
            ground_speed_kt=ac.ground_speed_kt,
            fuel_kg=ac.fuel_kg,
            fuel_burn_rate_kg_per_min=ac.fuel_burn_rate_kg_per_min,
            reserve_fuel_kg=ac.reserve_fuel_kg,
            emergency_flag=True,
            emergency_type=emg_type,
            runway_needed=(ac.runway_needed or near_rwy
                           or next(iter(sector.runway_availability), None)),
            vertical_speed_fpm=ac.vertical_speed_fpm,
        )

    st.session_state.sector = SectorState(
        aircraft=[_patch(ac) for ac in sector.aircraft],
        runway_availability=dict(sector.runway_availability),
        sim_time_s=sector.sim_time_s,
        wind_north_kt=sector.wind_north_kt,
        wind_east_kt=sector.wind_east_kt,
    )
    st.session_state.needs_recompute = True


# ══════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_radar_fig(sector, active_action=None) -> go.Figure:
    """
    Radar plot with:
      - 5-min dashed projected paths for all aircraft
      - Collision-aware label positions
      - Visual triage: emergency (large red) / critical (amber) / normal (green-grad)
      - Airport markers with city/country annotations
      - When active_action is set: ghost + amber reroute line + animated dot
        (AssignRunway), new heading line (Vector), or hold orbit (Hold)
    """
    from cascade_engine import AssignRunwayAction, HoldAction, VectorAction

    try:
        from cascade_engine import _check_separation as _sep_check
    except Exception:
        _sep_check = None

    fig         = go.Figure()
    annotations = []                 # collected and added via update_layout
    shapes      = []                 # static scope furniture (rings, radials, face)

    # ── scope geometry — centre + radius drive the range rings ─────────────
    all_lats = ([ac.lat for ac in sector.aircraft]
                + [ap['lat'] for ap in _AIRPORTS.values()])
    all_lons = ([ac.lon for ac in sector.aircraft]
                + [ap['lon'] for ap in _AIRPORTS.values()])
    lon_min, lon_max = min(all_lons) - 1.0, max(all_lons) + 1.0
    lat_min, lat_max = min(all_lats) - 0.6, max(all_lats) + 0.6
    cx, cy  = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    ratio   = 1.6   # yaxis scaleratio — also used to keep rings screen-circular
    scope_r = math.hypot(lon_max - cx, (lat_max - cy) * ratio)

    def _circ(x0, y0, rx, n=56):
        """Screen-circular ring in data coords (corrects for the y scaleratio)."""
        return (
            [x0 + rx * math.cos(2 * math.pi * k / n) for k in range(n + 1)],
            [y0 + (rx / ratio) * math.sin(2 * math.pi * k / n) for k in range(n + 1)],
        )

    # ── scope face (vignette) + concentric range rings + bearing spokes ────
    shapes.append(dict(
        type='circle', xref='x', yref='y', layer='below',
        x0=cx - scope_r, x1=cx + scope_r,
        y0=cy - scope_r / ratio, y1=cy + scope_r / ratio,
        fillcolor=_RADAR_FACE, line=dict(color='rgba(99,160,220,0.22)', width=1),
    ))
    for frac in (0.28, 0.52, 0.76):
        rr = scope_r * frac
        shapes.append(dict(
            type='circle', xref='x', yref='y', layer='below',
            x0=cx - rr, x1=cx + rr, y0=cy - rr / ratio, y1=cy + rr / ratio,
            fillcolor='rgba(0,0,0,0)', line=dict(color=_RING_COL, width=1),
        ))
    for deg in range(0, 360, 30):
        a = math.radians(deg)
        shapes.append(dict(
            type='line', xref='x', yref='y', layer='below',
            x0=cx, y0=cy,
            x1=cx + scope_r * math.cos(a),
            y1=cy + (scope_r / ratio) * math.sin(a),
            line=dict(color=_RADIAL_COL, width=1),
        ))

    # ── conflicting aircraft (geometric ICAO check) drive the alert rings ──
    conflict_ids: set[str] = set()
    if _sep_check is not None:
        try:
            for b in _sep_check(sector):
                conflict_ids.add(b.aircraft_id_1)
                conflict_ids.add(b.aircraft_id_2)
        except Exception:
            pass

    _LBL_OFFSETS = [(26, -20), (26, 20), (-26, -20), (-26, 20)]
    _rank = {ac.id: i for i, ac in enumerate(
        sorted(sector.aircraft, key=lambda a: (a.lat, a.lon)))}

    # ── per-aircraft rendering ─────────────────────────────────────────────
    for ac in sector.aircraft:
        is_emg  = ac.emergency_flag
        is_crit = ac.is_fuel_critical and not is_emg

        if is_emg:
            col, glyph_sz, glow = _RED,        25, 'rgba(248,81,73,0.22)'
        elif is_crit:
            col, glyph_sz, glow = _ORANGE,     22, 'rgba(227,161,53,0.18)'
        else:
            col, glyph_sz, glow = _PLANE_NORM, 19, 'rgba(205,216,230,0.10)'

        # Trajectory fan — translucent cone of projected-path uncertainty
        fc_lat, fc_lon = _project_position(ac.lat, ac.lon, ac.heading_deg,
                                           ac.ground_speed_kt, 7.0)
        fl_lat, fl_lon = _project_position(ac.lat, ac.lon, ac.heading_deg - 8.5,
                                           ac.ground_speed_kt, 7.0)
        fr_lat, fr_lon = _project_position(ac.lat, ac.lon, ac.heading_deg + 8.5,
                                           ac.ground_speed_kt, 7.0)
        fig.add_trace(go.Scatter(
            x=[ac.lon, fl_lon, fc_lon, fr_lon, ac.lon],
            y=[ac.lat, fl_lat, fc_lat, fr_lat, ac.lat],
            mode='lines', fill='toself',
            fillcolor='rgba(190,200,215,0.06)',
            line=dict(color='rgba(190,200,215,0.10)', width=0.5),
            showlegend=False, hoverinfo='skip',
        ))

        # 5-min projected path (thin leader)
        p_lat, p_lon = _project_position(
            ac.lat, ac.lon, ac.heading_deg, ac.ground_speed_kt, 5.0
        )
        fig.add_trace(go.Scatter(
            x=[ac.lon, p_lon], y=[ac.lat, p_lat],
            mode='lines', line=dict(color=col, width=1.2),
            opacity=0.5, showlegend=False, hoverinfo='skip',
        ))

        # Conflict / emergency alert rings — nested translucent glow
        if ac.id in conflict_ids or is_emg:
            rc = (248, 81, 73) if is_emg else (227, 161, 53)
            for rr, op in ((scope_r * 0.070, 0.05),
                           (scope_r * 0.050, 0.09),
                           (scope_r * 0.032, 0.14)):
                rx, ry = _circ(ac.lon, ac.lat, rr)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry, mode='lines', fill='toself',
                    fillcolor=f'rgba({rc[0]},{rc[1]},{rc[2]},{op})',
                    line=dict(color=f'rgba({rc[0]},{rc[1]},{rc[2]},0.35)', width=0.8),
                    showlegend=False, hoverinfo='skip',
                ))

        # Soft glow halo — doubles as the hover target for the aircraft
        emg_line  = f"<br>🚨 <b>{ac.emergency_type}</b>" if is_emg else ""
        crit_line = "  ⚠ FUEL CRITICAL" if is_crit else ""
        hover = (
            f"<b>{ac.id}</b>{emg_line}{crit_line}<br>"
            f"FL{ac.altitude_ft/100:.0f}  {ac.heading_deg:.0f}°  "
            f"{ac.ground_speed_kt:.0f} kt<br>"
            f"Fuel {ac.fuel_kg:.0f} kg  "
            f"({ac.fuel_minutes_above_reserve:.0f} min above reserve)<br>"
            f"Burn {ac.fuel_burn_rate_kg_per_min:.1f} kg/min<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[ac.lon], y=[ac.lat], mode='markers',
            marker=dict(size=glyph_sz + 12, color=glow, line=dict(width=0)),
            hovertemplate=hover, showlegend=False,
        ))

        # Rotated aircraft glyph (✈ points NE by default → offset heading by 45°)
        annotations.append(dict(
            x=ac.lon, y=ac.lat, text='✈',
            showarrow=False, textangle=ac.heading_deg - 45,
            font=dict(color=col, size=glyph_sz),
        ))

        # Callsign + ETA tag with a faint leader line back to the aircraft
        ap_icao, _r = _nearest_airport_runway(sector, ac)
        eta_txt = ''
        if ap_icao and ap_icao in _AIRPORTS:
            apd  = _AIRPORTS[ap_icao]
            d_nm = math.hypot(apd['lat'] - ac.lat, apd['lon'] - ac.lon) * 60
            eta_txt = f": {d_nm / max(ac.ground_speed_kt, 1) * 60:.0f} min"
        ax_off, ay_off = _LBL_OFFSETS[_rank[ac.id] % 4]
        annotations.append(dict(
            x=ac.lon, y=ac.lat, text=f"{ac.id}{eta_txt}",
            showarrow=True, arrowhead=0, arrowwidth=0.7,
            arrowcolor='rgba(160,180,210,0.40)',
            ax=ax_off, ay=ay_off,
            font=dict(color=col, size=9.5),
            bgcolor='rgba(7,16,30,0.70)',
            bordercolor='rgba(99,160,220,0.20)', borderwidth=1, borderpad=2,
            xanchor='left' if ax_off > 0 else 'right',
        ))

    # ── airport markers (subtle — must not compete with aircraft) ──────────
    for icao, ap in _AIRPORTS.items():
        rwy_str = ', '.join(ap['runways']) if ap['runways'] else '—'
        fig.add_trace(go.Scatter(
            x=[ap['lon']], y=[ap['lat']], mode='markers',
            marker=dict(size=11, symbol='square', color='rgba(10,22,40,0.9)',
                        line=dict(color='rgba(88,166,255,0.6)', width=1.5)),
            showlegend=False,
            hovertemplate=(f"<b>{icao}</b>  {ap['city']}, {ap['country']}<br>"
                           f"Runways: {rwy_str}<extra></extra>"),
        ))
        annotations.append(dict(
            x=ap['lon'], y=ap['lat'],
            text=(f"<span style='color:#7d8fa6;font-size:10px;'>"
                  f"<b>{icao}</b> · {ap['city']}</span>"),
            showarrow=False, yshift=15, align='center',
        ))

    # ── reroute paths (BUG 2): original (dashed) + rerouted (solid) ────────
    # Fully data-driven from the persistent assignment + frozen original-path
    # snapshot, so BOTH lines stay visible for every reassigned aircraft —
    # independent of which single action is currently "active".
    assignments    = st.session_state.get("assignments", {})
    original_paths = st.session_state.get("original_paths", {})
    diversions     = st.session_state.get("diversions", {})
    sector_by_id   = {a.id: a for a in sector.aircraft}
    has_original_path = False
    has_reroute       = False

    for cs, rwy_new in assignments.items():
        ac = sector_by_id.get(cs)
        if ac is None:
            continue
        # Diverted (bumped) flights are drawn amber so the displacement stands out.
        sev_col = (_RED if ac.emergency_flag
                   else _ORANGE if (cs in diversions or ac.is_fuel_critical)
                   else _PLANE_NORM)

        # Original planned path — frozen at declaration (dashed)
        op = original_paths.get(cs)
        if op and op.get("airport") in _AIRPORTS:
            oap = _AIRPORTS[op["airport"]]
            fig.add_trace(go.Scatter(
                x=[op["from_lon"], oap['lon']], y=[op["from_lat"], oap['lat']],
                mode='lines', line=dict(color='#8b949e', width=2, dash='dash'),
                showlegend=False,
                hovertemplate=(f"<b>{cs}</b> original plan<br>"
                               f"→ {op['airport']} RWY {op['runway'] or '?'}"
                               f"<extra></extra>"),
            ))
            has_original_path = True

        # Rerouted path — current position → assigned runway threshold (solid)
        icao_new = _RUNWAY_AIRPORT.get(rwy_new)
        nap      = _AIRPORTS.get(icao_new) if icao_new else None
        if nap:
            d_nm = math.hypot(nap['lat'] - ac.lat, nap['lon'] - ac.lon) * 60
            eta  = d_nm / max(ac.ground_speed_kt, 1) * 60
            fig.add_trace(go.Scatter(
                x=[ac.lon, nap['lon']], y=[ac.lat, nap['lat']],
                mode='lines', line=dict(color=sev_col, width=2.5),
                showlegend=False,
                hovertemplate=(f"<b>{cs}</b> rerouted<br>"
                               f"→ {icao_new} RWY {rwy_new}<br>"
                               f"ETA ≈ {eta:.0f} min · {d_nm:.0f} NM<extra></extra>"),
            ))
            has_reroute = True

    # ── action overlay (landing marker + holds + ping-pong animation) ──────
    animated_trace_idx = None
    anim_start = anim_end = (0.0, 0.0)

    if isinstance(active_action, AssignRunwayAction):
        tgt_ac  = next((a for a in sector.aircraft
                        if a.id == active_action.aircraft_id), None)
        # Honour the LOCKED assignment so the animation + landing marker target the
        # exact runway the aircraft is flying toward (map ↔ table agreement).
        runway_id = assignments.get(active_action.aircraft_id, active_action.runway_id)
        ap_icao   = _RUNWAY_AIRPORT.get(runway_id)
        tgt_ap    = _AIRPORTS.get(ap_icao) if ap_icao else None

        if tgt_ac and tgt_ap:
            is_mayday = (tgt_ac.emergency_flag
                         and tgt_ac.emergency_type == "MAYDAY")
            approach_col = '#00d4ff' if is_mayday else _ORANGE  # cyan for MAYDAY

            dist_nm = (math.sqrt((tgt_ap['lat'] - tgt_ac.lat)**2
                                 + (tgt_ap['lon'] - tgt_ac.lon)**2) * 60)
            eta_min = dist_nm / max(tgt_ac.ground_speed_kt, 1) * 60

            # Landing marker at the assigned runway with ETA annotation
            fig.add_trace(go.Scatter(
                x=[tgt_ap['lon']], y=[tgt_ap['lat']], mode='markers',
                name='Landed',
                marker=dict(size=14, symbol='star', color=approach_col,
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hovertemplate=(f"<b>{ap_icao}</b> RWY {runway_id}<br>"
                               f"{tgt_ac.id} ETA ≈ {eta_min:.0f} min<extra></extra>"),
            ))
            annotations.append(dict(
                x=tgt_ap['lon'], y=tgt_ap['lat'],
                text=f"<b>LAND {eta_min:.0f}m</b>",
                showarrow=True, arrowhead=2,
                arrowcolor=approach_col, ax=20, ay=-30,
                font=dict(color=approach_col, size=10),
                bgcolor='rgba(13,17,23,0.85)',
            ))

            # Hold orbits + delay labels for non-emergency aircraft displaced
            # by the priority approach (within 80 NM)
            for other_ac in sector.aircraft:
                if other_ac.id == tgt_ac.id or other_ac.emergency_flag or other_ac.is_fuel_critical:
                    continue
                d_nm = (math.sqrt((other_ac.lat - tgt_ap['lat'])**2
                                  + (other_ac.lon - tgt_ap['lon'])**2) * 60)
                if d_nm < 80.0:  # within priority approach radius (80 NM)
                    r_deg   = 0.06
                    cos_lat = max(math.cos(math.radians(other_ac.lat)), 1e-9)
                    thetas  = [i * 2 * math.pi / 24 for i in range(25)]
                    fig.add_trace(go.Scatter(
                        x=[other_ac.lon + r_deg * math.sin(t) / cos_lat for t in thetas],
                        y=[other_ac.lat + r_deg * math.cos(t) for t in thetas],
                        mode='lines',
                        line=dict(color=_ORANGE, width=1.5, dash='dot'),
                        showlegend=False, hoverinfo='skip',
                    ))
                    annotations.append(dict(
                        x=other_ac.lon, y=other_ac.lat,
                        text=f"<b>+{eta_min:.0f}m delay</b>",
                        showarrow=True, arrowhead=0,
                        arrowcolor=_ORANGE, ax=0, ay=-35,
                        font=dict(color=_ORANGE, size=10),
                        bgcolor='rgba(13,17,23,0.85)',
                    ))

            # Animated dot pinging along the rerouted approach (frames update
            # only this trace — see below)
            anim_start = (tgt_ac.lat, tgt_ac.lon)
            anim_end   = (tgt_ap['lat'], tgt_ap['lon'])
            animated_trace_idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon], y=[tgt_ac.lat], mode='markers',
                marker=dict(size=18, color=approach_col,
                            line=dict(color='white', width=2.5)),
                showlegend=False,
                hovertemplate=(f"<b>{tgt_ac.id}</b> → {ap_icao}<br>"
                               f"Runway {runway_id}<br>"
                               f"ETA ≈ {eta_min:.0f} min<extra></extra>"),
            ))

    elif isinstance(active_action, VectorAction):
        tgt_ac = next((a for a in sector.aircraft
                       if a.id == active_action.aircraft_id), None)
        if tgt_ac:
            p_lat, p_lon = _project_position(
                tgt_ac.lat, tgt_ac.lon,
                active_action.new_heading_deg, tgt_ac.ground_speed_kt, 5.0
            )
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon, p_lon], y=[tgt_ac.lat, p_lat],
                mode='lines', line=dict(color=_ORANGE, width=2.5),
                showlegend=False, hoverinfo='skip',
            ))

    elif active_action and isinstance(active_action, HoldAction):
        tgt_ac = next((a for a in sector.aircraft
                       if a.id == active_action.aircraft_id), None)
        if tgt_ac:
            r_deg   = 0.08
            cos_lat = max(math.cos(math.radians(tgt_ac.lat)), 1e-9)
            thetas  = [i * 2 * math.pi / 24 for i in range(25)]
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon + r_deg * math.sin(t) / cos_lat for t in thetas],
                y=[tgt_ac.lat + r_deg * math.cos(t) for t in thetas],
                mode='lines', line=dict(color=_ORANGE, width=2, dash='dot'),
                showlegend=False, hoverinfo='skip',
            ))

    # ── animation frames (ping-pong) ──────────────────────────────────────
    has_anim = animated_trace_idx is not None
    if has_anim:
        N = 40
        half = N // 2
        s_lat, s_lon = anim_start
        e_lat, e_lon = anim_end
        frames = []
        for i in range(N):
            # 0..half-1 → forward (0→1), half..N-1 → backward (1→0)
            raw = i if i < half else N - 1 - i
            t   = max(0.0, min(1.0, raw / max(half - 1, 1)))
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=[s_lon + (e_lon - s_lon) * t],
                    y=[s_lat + (e_lat - s_lat) * t],
                )],
                traces=[animated_trace_idx],
                name=str(i),
            ))
        fig.frames = frames

        play_btn  = dict(
            label='▶ Play', method='animate',
            args=[None, dict(frame=dict(duration=80, redraw=True),
                             fromcurrent=True, mode='immediate')],
        )
        pause_btn = dict(
            label='⏸ Pause', method='animate',
            args=[[None], dict(frame=dict(duration=0, redraw=False),
                               mode='immediate')],
        )
        fig.update_layout(updatemenus=[dict(
            type='buttons', showactive=False,
            buttons=[play_btn, pause_btn],
            x=0.13, y=-0.05, xanchor='right', yanchor='top',
            bgcolor='#161b22', bordercolor='#30363d',
            font=dict(color=_TEXT, size=11),
        )])

    # ── legend annotation (restyled for the dark scope) ────────────────────
    path_legend = (
        "   <span style='color:#f85149'>━━</span> rerouted path"
        if has_reroute else ""
    )
    orig_legend = (
        "   <span style='color:#8b949e'>╌╌</span> original plan"
        if has_original_path else ""
    )
    annotations.append(dict(
        text=(
            "✈ <span style='color:#d7dde6'>normal</span>"
            "  ✈ <span style='color:#e3a135'>critical</span>"
            "  ✈ <span style='color:#f85149'>emergency</span>"
            "  ⬛ <span style='color:#7d8fa6'>airport</span><br>"
            "<span style='color:#5a6b80'>──</span> 5-min path"
            "   <span style='color:#9aa7b8'>◹</span> trajectory fan"
            + path_legend + orig_legend
        ),
        xref='paper', yref='paper', x=0.01, y=0.01,
        xanchor='left', yanchor='bottom', showarrow=False,
        bgcolor='rgba(6,14,28,0.85)', bordercolor='rgba(99,160,220,0.25)',
        borderwidth=1, borderpad=5,
        font=dict(color=_TEXT, size=9), align='left',
    ))

    # Scale note
    annotations.append(dict(
        text='1° lat ≈ 60 nm',
        xref='paper', yref='paper', x=0.99, y=0.01,
        xanchor='right', yanchor='bottom', showarrow=False,
        font=dict(color='#3f4b5b', size=9),
    ))

    # ── layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        showlegend=False,
        plot_bgcolor=_RADAR_BG, paper_bgcolor=_RADAR_BG,
        height=490,
        margin=dict(l=10, r=10, t=30, b=55 if has_anim else 10),
        shapes=shapes,
        title=dict(
            text=(f"Sector radar  ·  t = {sector.sim_time_s:.0f} s  "
                  f"·  {len(sector.aircraft)} aircraft"),
            font=dict(color=_TEXT, size=13), x=0.02,
        ),
        xaxis=dict(
            range=[lon_min, lon_max],
            showgrid=False, zeroline=False,
            tickfont=dict(color='#4d6b8a', size=9),
            title=dict(text='Longitude °E', font=dict(color='#5f7d9a', size=11)),
        ),
        yaxis=dict(
            range=[lat_min, lat_max],
            showgrid=False, zeroline=False,
            tickfont=dict(color='#4d6b8a', size=9),
            title=dict(text='Latitude °N', font=dict(color='#5f7d9a', size=11)),
            scaleanchor='x', scaleratio=ratio,
        ),
        annotations=annotations,
    )
    return fig


def _make_fuel_fig(sector) -> go.Figure:
    acs  = sorted(sector.aircraft, key=lambda a: a.fuel_minutes_above_reserve)
    ids  = [ac.id for ac in acs]
    mins = [min(ac.fuel_minutes_above_reserve, 300) for ac in acs]
    cols = [_RED if ac.emergency_flag else _ORANGE if ac.is_fuel_critical else _GREEN
            for ac in acs]

    fig = go.Figure(go.Bar(
        y=ids, x=mins, orientation="h",
        marker_color=cols,
        text=[f"{m:.0f} min" for m in mins],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))
    fig.add_vline(x=30, line_dash="dash", line_color=_RED, line_width=1.5,
                  annotation_text="Reserve floor", annotation_font_color=_RED,
                  annotation_position="top right")
    fig.update_layout(
        height=200, margin=dict(l=10, r=60, t=20, b=10),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis=dict(title="Minutes above reserve", gridcolor=_GRID,
                   tickfont=dict(color=_GREEN), range=[0, max(mins) * 1.15 + 10]),
        yaxis=dict(tickfont=dict(color=_TEXT)),
    )
    return fig


def _make_timeline_fig(do_nothing_score, action_score) -> go.Figure:
    """Two-row timeline: top = do-nothing, bottom = recommended action."""
    horizon   = action_score.horizon_s
    interval  = action_score.checkpoint_interval_s
    n         = int(horizon / interval)
    checkpoints = [k * interval for k in range(1, n + 1)]

    def _row_traces(score, row_label: str, y_offset: float):
        fuel_t = {b.elapsed_s for b in score.fuel_breaches}
        sep_t  = {b.elapsed_s for b in score.separation_breaches}
        traces, annotations = [], []
        for i, t in enumerate(checkpoints):
            t_start = checkpoints[i-1] if i > 0 else 0
            if t in sep_t:
                col, tip = _RED,    f"SEP  t+{t:.0f}s"
            elif t in fuel_t:
                col, tip = _ORANGE, f"FUEL t+{t:.0f}s"
            else:
                col, tip = _GREEN,  f"OK   t+{t:.0f}s"
            traces.append(go.Bar(
                x=[t - t_start], y=[row_label], orientation="h",
                base=t_start, marker_color=col, marker_opacity=0.75,
                showlegend=False,
                hovertemplate=tip + "<extra></extra>",
            ))
        # Breach annotations (only on first detection)
        for b in score.fuel_breaches:
            annotations.append(dict(
                x=b.elapsed_s, y=y_offset + 0.55,
                text=f"⛽{b.aircraft_id}", showarrow=True,
                arrowhead=2, arrowcolor=_ORANGE,
                font=dict(color=_ORANGE, size=9),
            ))
        for b in score.separation_breaches:
            annotations.append(dict(
                x=b.elapsed_s, y=y_offset + 0.55,
                text=f"⚠{b.aircraft_id_1}/{b.aircraft_id_2}",
                showarrow=True, arrowhead=2, arrowcolor=_RED,
                font=dict(color=_RED, size=9),
            ))
        return traces, annotations

    fig = go.Figure()
    t1, a1 = _row_traces(do_nothing_score, "Do nothing", 1)
    t2, a2 = _row_traces(action_score,     "Recommended", 0)
    for t in t1 + t2:
        fig.add_trace(t)

    fig.update_layout(
        annotations=a1 + a2,
        height=140, margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        barmode="stack",
        xaxis=dict(
            range=[0, horizon],
            tickvals=list(range(0, int(horizon)+1, 60)),
            ticktext=[f"{int(v/60)}m" for v in range(0, int(horizon)+1, 60)],
            tickfont=dict(color=_GREEN), gridcolor=_GRID,
            title=dict(text="Lookahead", font=dict(color=_GREEN, size=10)),
        ),
        yaxis=dict(tickfont=dict(color=_TEXT), categoryorder="array",
                   categoryarray=["Recommended", "Do nothing"]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI component renderers
# ══════════════════════════════════════════════════════════════════════════════

def _action_english(action) -> str:
    from cascade_engine import NoAction, AssignRunwayAction, HoldAction, VectorAction
    if isinstance(action, NoAction):
        return "Take no action — sector is stable"
    if isinstance(action, AssignRunwayAction):
        airport = _RUNWAY_AIRPORT.get(action.runway_id)
        location = f" at {airport}" if airport else ""
        return f"Assign **{action.aircraft_id}** to runway **{action.runway_id}**{location} (priority approach)"
    if isinstance(action, HoldAction):
        return f"Place **{action.aircraft_id}** in a holding pattern"
    if isinstance(action, VectorAction):
        return f"Vector **{action.aircraft_id}** to heading **{action.new_heading_deg:.0f}°**"
    return str(action)


def _render_recommendation_card():
    sx_reco   = st.session_state.get("sx_reco")
    bl_action = st.session_state.get("bl_action")
    bl_score  = st.session_state.get("bl_score")
    mode      = st.session_state.mode

    if sx_reco is None:
        st.caption("Computing recommendation…")
        return

    primary_action = sx_reco.action if mode == "SKYLANCE-X" else bl_action
    primary_score  = sx_reco.score  if mode == "SKYLANCE-X" else bl_score

    breaches    = primary_score.total_breach_count
    status_col  = _RED if breaches > 0 else _GREEN
    status_dot  = f"<span style='display:inline-block;width:7px;height:7px;" \
                  f"border-radius:50%;background:{status_col};margin-right:6px;" \
                  f"vertical-align:middle;'></span>"
    breach_txt  = (
        f"{breaches} conflict{'s' if breaches != 1 else ''} — "
        f"first at t+{primary_score.time_to_first_breach_s:.0f}s"
        if breaches > 0 else "No conflicts in lookahead window"
    )

    st.markdown(f"""
<div style="border-left:2px solid #30363d;padding:10px 14px;margin-bottom:10px;">
  <div style="font-size:11px;color:{_MUTED};margin-bottom:6px;">
    {mode} &mdash; recommended action
  </div>
  <div style="font-size:14px;color:#e6edf3;font-weight:500;line-height:1.5;">
    {_action_english(primary_action)}
  </div>
  <div style="font-size:12px;color:{status_col};margin-top:8px;">
    {status_dot}{breach_txt}
  </div>
</div>
""", unsafe_allow_html=True)

    # Compact side-by-side breach count comparison
    sx_b = sx_reco.score.total_breach_count
    bl_b = bl_score.total_breach_count if bl_score else None
    c1, c2 = st.columns(2)
    with c1:
        col = _GREEN if sx_b == 0 else _ORANGE if sx_b < 2 else _RED
        st.markdown(
            f"<div style='font-size:11px;color:{_MUTED};'>SKYLANCE-X</div>"
            f"<div style='font-size:16px;color:{col};font-weight:600;'>"
            f"{sx_b} conflict{'s' if sx_b != 1 else ''}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        if bl_b is not None:
            col = _GREEN if bl_b == 0 else _ORANGE if bl_b < 2 else _RED
            st.markdown(
                f"<div style='font-size:11px;color:{_MUTED};'>Baseline</div>"
                f"<div style='font-size:16px;color:{col};font-weight:600;'>"
                f"{bl_b} conflict{'s' if bl_b != 1 else ''}</div>",
                unsafe_allow_html=True,
            )


def _render_conformal_ribbon():
    cert = st.session_state.get("certificate")
    if cert is None:
        return

    if cert.certified_safe:
        label   = f"Certified safe &mdash; {cert.confidence_pct:.0f}% confidence"
        detail  = f"nc {cert.nc:.3f} ≤ {cert.threshold:.3f}  ·  p = {cert.p_value:.3f}"
        lcolor  = _GREEN
    else:
        label   = "Certification failed &mdash; separation risk detected"
        detail  = f"nc {cert.nc:.3f} > {cert.threshold:.3f}  ·  p = {cert.p_value:.3f}"
        lcolor  = _RED

    st.markdown(f"""
<div style="border-left:2px solid {lcolor};padding:7px 12px;margin:8px 0 4px;">
  <div style="font-size:13px;color:{lcolor};font-weight:500;">{label}</div>
  <div style="font-size:11px;color:{_MUTED};margin-top:2px;">{detail}</div>
</div>
""", unsafe_allow_html=True)


def _render_cascade_strip():
    sx_reco = st.session_state.get("sx_reco")
    if sx_reco is None:
        return

    do_nothing = sx_reco.baseline_score
    action_sc  = (sx_reco.score if st.session_state.mode == "SKYLANCE-X"
                  else st.session_state.get("bl_score", sx_reco.score))

    def _summary(score, label: str) -> str:
        if score.total_breach_count == 0:
            return f"**{label}** — clear"
        parts = []
        for b in score.fuel_breaches:
            parts.append(f"{b.aircraft_id} fuel at t+{b.elapsed_s/60:.0f}m")
        for b in score.separation_breaches:
            parts.append(f"{b.aircraft_id_1}/{b.aircraft_id_2} sep at "
                         f"t+{b.elapsed_s/60:.0f}m")
        return f"**{label}** — " + ", ".join(parts)

    st.caption(_summary(do_nothing, "No action"))
    st.caption(_summary(action_sc,  "Recommended"))
    st.plotly_chart(
        _make_timeline_fig(do_nothing, action_sc),
        use_container_width=True,
    )


def _render_counterfactual():
    cf = st.session_state.get("counterfactual")
    if cf is None:
        st.caption("Use 'Explain decision' in the sidebar for a minimal-flip explanation.")
        return
    st.markdown(f"""
<div style="border-left:2px solid #30363d;padding:8px 12px;margin-top:6px;">
  <div style="font-size:11px;color:{_MUTED};margin-bottom:4px;">Why this action?</div>
  <div style="font-size:13px;color:#c9d1d9;line-height:1.55;">{cf.sentence}</div>
  <div style="font-size:11px;color:#484f58;margin-top:4px;">
    feature: <code>{cf.feature}</code> &nbsp;·&nbsp; cost {cf.normalized_cost:.3f}
  </div>
</div>
""", unsafe_allow_html=True)


def _render_decisions_panel():
    from cascade_engine import AssignRunwayAction
    sector = st.session_state.sector
    sx_reco = st.session_state.get("sx_reco")
    if sx_reco is None:
        return

    mode = st.session_state.get("mode", "SKYLANCE-X")
    active_action = sx_reco.action if mode == "SKYLANCE-X" else st.session_state.get("bl_action")

    if not isinstance(active_action, AssignRunwayAction):
        return

    tgt_ac = next((a for a in sector.aircraft if a.id == active_action.aircraft_id), None)
    if not tgt_ac or not (tgt_ac.emergency_flag and tgt_ac.emergency_type == "MAYDAY"):
        return

    ap_icao = _RUNWAY_AIRPORT.get(active_action.runway_id)
    tgt_ap = _AIRPORTS.get(ap_icao) if ap_icao else None
    if not tgt_ap:
        return

    dist_nm = (math.sqrt((tgt_ap['lat'] - tgt_ac.lat)**2
                         + (tgt_ap['lon'] - tgt_ac.lon)**2) * 60)
    eta_min = dist_nm / max(tgt_ac.ground_speed_kt, 1) * 60

    st.markdown(
        "<div style='font-size:12px;color:#6e7681;margin:14px 0 4px;'>"
        "Priority Approach Decisions</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
<div style="background-color:rgba(248,81,73,0.07);border:1px solid rgba(248,81,73,0.25);border-radius:6px;padding:12px;margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-weight:600;color:{_RED};font-size:14px;">🚨 {tgt_ac.id} (MAYDAY)</span>
        <span style="background-color:{_RED};color:white;font-size:10px;font-weight:bold;padding:2px 6px;border-radius:10px;">PRIORITY LANDING</span>
    </div>
    <div style="margin-top:8px;font-size:12px;color:#c9d1d9;line-height:1.6;">
        <b>Airport:</b> {ap_icao} ({tgt_ap['city']})<br>
        <b>Runway:</b> {active_action.runway_id}<br>
        <b>ETA:</b> {eta_min:.1f} min ({dist_nm:.1f} NM)<br>
        <b>Status:</b> Cleared for straight-in approach.
    </div>
</div>
""", unsafe_allow_html=True)

    # Diverted flights — bumped to a different airport because runways are full
    diversions     = st.session_state.get("diversions", {})
    original_paths = st.session_state.get("original_paths", {})
    div_rows = []
    for cs, (alt_icao, alt_rwy) in diversions.items():
        a = next((x for x in sector.aircraft if x.id == cs), None)
        if a is None:
            continue
        alt_ap  = _AIRPORTS.get(alt_icao)
        eta_div = "—"
        if alt_ap:
            dd = math.hypot(alt_ap['lat'] - a.lat, alt_ap['lon'] - a.lon) * 60
            eta_div = f"{dd / max(a.ground_speed_kt, 1) * 60:.0f} min"
        from_ap = original_paths.get(cs, {}).get("airport") or ap_icao
        div_rows.append({
            "Callsign":    cs,
            "Was":         from_ap or "—",
            "Diverted to": f"{alt_icao} · {alt_rwy}",
            "ETA":         eta_div,
        })

    if div_rows:
        st.markdown(
            f"<div style='font-size:11px;color:{_ORANGE};margin-bottom:6px;'>"
            f"↪ Diverted Flights — runways full at {ap_icao}</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(div_rows, use_container_width=True, hide_index=True,
                     height=min(200, 35 + 35 * len(div_rows)))

    # Remaining nearby non-emergency aircraft that still hold/sequence (not diverted)
    affected_rows = []
    for other_ac in sector.aircraft:
        if (other_ac.id == tgt_ac.id or other_ac.emergency_flag
                or other_ac.is_fuel_critical or other_ac.id in diversions):
            continue
        d_nm = (math.sqrt((other_ac.lat - tgt_ap['lat'])**2
                          + (other_ac.lon - tgt_ap['lon'])**2) * 60)
        if d_nm < 80.0:  # within priority approach radius (80 NM)
            affected_rows.append({
                "Callsign": other_ac.id,
                "Decision": "HOLD (Orbit)",
                "Distance": f"{d_nm:.1f} NM",
                "Delay": f"+{eta_min:.0f} min",
            })

    if affected_rows:
        st.markdown(f"""
<div style="font-size:11px;color:{_MUTED};margin-bottom:6px;">Affected Aircraft in Hold Sector (80 NM)</div>
""", unsafe_allow_html=True)
        st.dataframe(
            affected_rows,
            use_container_width=True,
            hide_index=True,
            height=min(200, 35 + 35 * len(affected_rows))
        )
    elif not div_rows:
        st.markdown(f"""
<div style="font-size:11px;color:{_MUTED};margin-bottom:6px;">No other aircraft affected in the approach sector.</div>
""", unsafe_allow_html=True)


def _render_landing_plan_comparison():
    baseline_plan   = st.session_state.get("baseline_plan")
    current_plan    = st.session_state.get("current_plan")
    baseline_locked = st.session_state.get("baseline_locked", False)

    st.markdown(
        "<div style='font-size:12px;color:#6e7681;margin:14px 0 4px;'>"
        "Landing Plan Comparison</div>",
        unsafe_allow_html=True,
    )

    if not baseline_locked:
        st.markdown(
            f"<div style='font-size:12px;color:{_MUTED};padding:8px 12px;"
            "border:1px solid #30363d;border-radius:4px;'>"
            "No emergency active — baseline plan only.</div>",
            unsafe_allow_html=True,
        )
        return

    if baseline_plan is None:
        st.markdown(
            f"<div style='font-size:12px;color:{_ORANGE};padding:8px 12px;"
            f"border:1px solid {_ORANGE};border-radius:4px;'>"
            "&#9888; Baseline snapshot missing — load a new sector before declaring an emergency.</div>",
            unsafe_allow_html=True,
        )
        return

    if current_plan is None:
        st.caption("Computing current plan…")
        return

    diffs = _diff_plans(baseline_plan, current_plan)
    b_map = {p["callsign"]: p for p in baseline_plan}
    c_map = {p["callsign"]: p for p in current_plan}

    def _fmt_eta(eta_min):
        return f"{eta_min:.1f} min" if eta_min is not None else "—"

    def _row_bg(status):
        if status == "priority":
            return f"background:rgba(248,81,73,0.12);border-left:3px solid {_RED};"
        if status in ("rerouted", "resequenced"):
            return f"background:rgba(227,161,53,0.10);border-left:3px solid {_ORANGE};"
        return "border-left:3px solid transparent;"

    th = (f"<th style='padding:4px 6px;font-size:10px;color:{_MUTED};"
          "text-align:left;font-weight:500;'>")

    def _table(rows_html):
        return (
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>{th}Callsign</th>{th}Airport</th>"
            f"{th}Runway</th>{th}ETA</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>"
        )

    # ── Before column ─────────────────────────────────────────────────────────
    td      = "style='padding:5px 6px;font-size:12px;"
    td_hdr  = td + "color:#e6edf3;'"
    td_norm = td + "color:" + _TEXT + ";'"

    before_rows = ""
    for d in diffs:
        b = b_map.get(d["callsign"])
        if not b:
            continue
        row_style = _row_bg(d["status"])
        callsign  = d["callsign"]
        emg_type  = d.get("emergency_type") or ""
        badge = ""
        if d["is_emergency"] and emg_type:
            badge = (
                "<span style='font-size:9px;background:" + _RED + ";color:white;"
                "border-radius:3px;padding:1px 4px;margin-left:4px;'>"
                + emg_type + "</span>"
            )
        b_ap  = b["assigned_airport"] or "—"
        b_rwy = b["assigned_runway"]  or "—"
        b_eta = _fmt_eta(b["eta_min"])
        before_rows += (
            "<tr style='" + row_style + "'>"
            "<td " + td_hdr + "><b>" + callsign + "</b>" + badge + "</td>"
            "<td " + td_norm + ">" + b_ap  + "</td>"
            "<td " + td_norm + ">" + b_rwy + "</td>"
            "<td " + td_norm + ">" + b_eta + "</td></tr>"
        )

    # ── After column ──────────────────────────────────────────────────────────
    after_rows = ""
    for d in diffs:
        c = c_map.get(d["callsign"])
        if not c:
            continue

        row_style = _row_bg(d["status"])
        callsign  = d["callsign"]
        emg_type  = d.get("emergency_type") or ""

        if d["status"] == "priority":
            badge = (
                "<span style='font-size:9px;background:" + _RED + ";color:white;"
                "border-radius:3px;padding:1px 4px;margin-left:4px;'>PRIORITY</span>"
            )
        elif d["is_emergency"] and emg_type:
            badge = (
                "<span style='font-size:9px;background:" + _ORANGE + ";color:white;"
                "border-radius:3px;padding:1px 4px;margin-left:4px;'>"
                + emg_type + "</span>"
            )
        else:
            badge = ""

        if d["runway_changed"]:
            old_r = d["old_runway"] or "—"
            new_r = d["new_runway"] or "—"
            rwy_cell = (
                "<span style='color:" + _MUTED + ";text-decoration:line-through;'>"
                + old_r + "</span> &#8594; "
                "<b style='color:" + _ORANGE + ";'>" + new_r + "</b>"
            )
        else:
            rwy_cell = c["assigned_runway"] or "—"

        if d["airport_changed"]:
            old_a = d["old_airport"] or "—"
            new_a = d["new_airport"] or "—"
            ap_cell = (
                "<span style='color:" + _MUTED + ";text-decoration:line-through;'>"
                + old_a + "</span> &#8594; "
                "<b style='color:" + _ORANGE + ";'>" + new_a + "</b>"
            )
        else:
            ap_cell = c["assigned_airport"] or "—"

        eta_cell = _fmt_eta(c["eta_min"])
        delta    = d["eta_delta_min"]
        if delta is not None and abs(delta) > 0.5:
            sign      = "+" if delta > 0 else ""
            delta_col = _RED if delta > 0 else _GREEN
            delta_str = "{:.1f}".format(delta)
            eta_cell += (
                "<span style='font-size:10px;color:" + delta_col + ";"
                "margin-left:3px;'>(" + sign + delta_str + ")</span>"
            )

        after_rows += (
            "<tr style='" + row_style + "'>"
            "<td " + td_hdr + "><b>" + callsign + "</b>" + badge + "</td>"
            "<td " + td + "'>" + ap_cell  + "</td>"
            "<td " + td + "'>" + rwy_cell + "</td>"
            "<td " + td + "'>" + eta_cell + "</td></tr>"
        )

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown(
            f"<div style='font-size:11px;color:{_MUTED};margin-bottom:5px;"
            "font-weight:500;'>Before emergency</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_table(before_rows), unsafe_allow_html=True)
    with col_a:
        st.markdown(
            f"<div style='font-size:11px;color:{_MUTED};margin-bottom:5px;"
            "font-weight:500;'>After SKYLANCE-X decision</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_table(after_rows), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar():
    with st.sidebar:
        # ── wordmark ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='padding:6px 0 18px;'>"
            "<div style='font-size:15px;font-weight:600;color:#e6edf3;"
            "letter-spacing:-0.2px;'>SKYLANCE-X</div>"
            "<div style='font-size:11px;color:#6e7681;margin-top:2px;'>"
            "ATC Decision Support</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── system toggle ─────────────────────────────────────────────────
        new_mode = st.radio(
            "View",
            ["SKYLANCE-X", "Baseline"],
            index=0 if st.session_state.mode == "SKYLANCE-X" else 1,
            horizontal=True,
        )
        if new_mode != st.session_state.mode:
            st.session_state.mode = new_mode

        st.divider()

        # ── scenario ──────────────────────────────────────────────────────
        st.markdown("<div style='font-size:12px;color:#6e7681;margin-bottom:8px;'>"
                    "Scenario</div>", unsafe_allow_html=True)
        seed = st.number_input("Seed", value=st.session_state.sector_seed,
                               step=1, min_value=0, label_visibility="collapsed")
        n_ac = st.slider("Aircraft", 4, 12, st.session_state.n_aircraft)

        col1, col2 = st.columns(2)
        if col1.button("Load", use_container_width=True):
            st.session_state.sector_seed = seed
            st.session_state.n_aircraft  = n_ac
            _load_new_sector(seed, n_ac)
        if col2.button("Advance 60 s", use_container_width=True):
            _step_60s()

        st.divider()

        # ── runways ───────────────────────────────────────────────────────
        st.markdown("<div style='font-size:12px;color:#6e7681;margin-bottom:8px;'>"
                    "Runways</div>", unsafe_allow_html=True)
        sector  = st.session_state.sector
        changed = False
        new_rwa = dict(sector.runway_availability)
        for icao, ap in _AIRPORTS.items():
            ap_runways = [r for r in ap['runways'] if r in sector.runway_availability]
            if not ap_runways:
                continue
            st.markdown(f"<div style='font-size:11px;color:#484f58;"
                        f"margin:6px 0 2px;'>{icao}</div>", unsafe_allow_html=True)
            for rwy in ap_runways:
                avail   = sector.runway_availability[rwy]
                new_val = st.checkbox(rwy, value=avail, key=f"rwy_{rwy}")
                if new_val != avail:
                    new_rwa[rwy] = new_val
                    changed = True
        unassigned = [r for r in sector.runway_availability if r not in _RUNWAY_AIRPORT]
        if unassigned:
            st.markdown("<div style='font-size:11px;color:#484f58;margin:6px 0 2px;'>"
                        "Other</div>", unsafe_allow_html=True)
            for rwy in unassigned:
                avail   = sector.runway_availability[rwy]
                new_val = st.checkbox(rwy, value=avail, key=f"rwy_{rwy}")
                if new_val != avail:
                    new_rwa[rwy] = new_val
                    changed = True
        if changed:
            from state_schema import SectorState
            st.session_state.sector = SectorState(
                aircraft=list(sector.aircraft),
                runway_availability=new_rwa,
                sim_time_s=sector.sim_time_s,
                wind_north_kt=sector.wind_north_kt,
                wind_east_kt=sector.wind_east_kt,
            )
            st.session_state.needs_recompute = True

        st.divider()

        # ── declare emergency ─────────────────────────────────────────────
        st.markdown("<div style='font-size:12px;color:#6e7681;margin-bottom:8px;'>"
                    "Declare emergency</div>", unsafe_allow_html=True)
        ac_ids   = [ac.id for ac in st.session_state.sector.aircraft]
        chosen   = st.selectbox("Aircraft", ac_ids, label_visibility="collapsed")
        emg_type = st.selectbox("Type", ["MAYDAY", "PAN-PAN", "FUEL", "MED"],
                                label_visibility="collapsed")
        if st.button("Declare", use_container_width=True):
            _inject_emergency(chosen, emg_type)

        if st.button("Explain decision", use_container_width=True,
                     help="Minimal-flip counterfactual (~30 s)"):
            st.session_state.needs_explain = True

        # ── status line ───────────────────────────────────────────────────
        s     = st.session_state.sector
        n_emg = sum(1 for ac in s.aircraft if ac.emergency_flag)
        n_crit= sum(1 for ac in s.aircraft if ac.is_fuel_critical)
        n_rwy = sum(1 for v in s.runway_availability.values() if v)
        st.markdown(
            f"<div style='font-size:11px;color:#484f58;margin-top:16px;'>"
            f"t = {s.sim_time_s:.0f} s &nbsp;·&nbsp; {len(s.aircraft)} aircraft"
            f"&nbsp;·&nbsp; {n_emg} emg &nbsp;·&nbsp; {n_crit} crit"
            f"&nbsp;·&nbsp; {n_rwy} rwy open</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    _ensure_initialized()
    _render_sidebar()

    if st.session_state.get("needs_explain"):
        st.session_state.needs_explain = False
        _run_explainer()

    if st.session_state.get("needs_recompute"):
        with st.spinner("Analysing sector…"):
            _recompute()

    sector = st.session_state.sector
    col_radar, col_panel = st.columns([3, 2], gap="large")

    with col_radar:
        sx_reco   = st.session_state.get("sx_reco")
        bl_action = st.session_state.get("bl_action")
        mode      = st.session_state.get("mode", "SKYLANCE-X")
        active_action = (
            (sx_reco.action if sx_reco else None)
            if mode == "SKYLANCE-X" else bl_action
        )
        st.plotly_chart(_make_radar_fig(sector, active_action),
                        use_container_width=True)
        st.plotly_chart(_make_fuel_fig(sector), use_container_width=True)

    with col_panel:
        _render_recommendation_card()
        _render_conformal_ribbon()

        st.markdown(
            "<div style='font-size:12px;color:#6e7681;margin:14px 0 4px;'>"
            "Lookahead</div>",
            unsafe_allow_html=True,
        )
        _render_cascade_strip()
        _render_counterfactual()
        _render_decisions_panel()
        _render_landing_plan_comparison()

        st.markdown(
            "<div style='font-size:12px;color:#6e7681;margin:14px 0 4px;'>"
            "Fleet</div>",
            unsafe_allow_html=True,
        )
        rows = []
        for ac in sorted(sector.aircraft,
                         key=lambda a: a.fuel_minutes_above_reserve):
            status = []
            if ac.emergency_flag:
                status.append(ac.emergency_type or "EMG")
            if ac.is_fuel_critical:
                status.append("CRITICAL")
            rows.append({
                "Callsign":  ac.id,
                "FL":        f"{ac.altitude_ft/100:.0f}",
                "Hdg":       f"{ac.heading_deg:.0f}°",
                "GS (kt)":   f"{ac.ground_speed_kt:.0f}",
                "Fuel (min)":f"{ac.fuel_minutes_above_reserve:.0f}",
                "Status":    ", ".join(status) if status else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True,
                     height=min(380, 35 + 35 * len(rows)))


if __name__ == "__main__":
    main()
