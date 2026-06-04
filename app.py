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
"""

import math
import sys
import os

import streamlit as st
import plotly.graph_objects as go

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
_ARROW_LEN = 0.13    # degrees — heading vector length on plot

_CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] { border-right: 1px solid #21262d; }
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
    'EGLL': {'lat': 51.4700, 'lon': -0.4543, 'runways': ['27L', '27R'],
             'city': 'London', 'country': 'GB'},
    'EHAM': {'lat': 52.3086, 'lon':  4.7639, 'runways': ['09L', '09R'],
             'city': 'Amsterdam', 'country': 'NL'},
    'EDDF': {'lat': 50.0379, 'lon':  8.5622, 'runways': [],
             'city': 'Frankfurt', 'country': 'DE'},
}
_RUNWAY_AIRPORT = {
    rwy: icao
    for icao, ap in _AIRPORTS.items()
    for rwy in ap['runways']
}


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


_LABEL_POS = ['top center', 'bottom center', 'top right', 'bottom left']

def _label_pos_map(aircraft) -> dict[str, str]:
    """Assign textposition so clustered aircraft labels don't overlap."""
    # Rank aircraft by (lat, lon) to give each a stable index in the cycle.
    ranked = sorted(range(len(aircraft)), key=lambda i: (aircraft[i].lat, aircraft[i].lon))
    rank   = {aircraft[ranked[i]].id: i for i in range(len(ranked))}
    result = {}
    for ac in aircraft:
        crowded = any(
            abs(ac.lat - o.lat) < 0.5 and abs(ac.lon - o.lon) < 0.5
            for o in aircraft if o.id != ac.id
        )
        result[ac.id] = _LABEL_POS[rank[ac.id] % 4] if crowded else 'top center'
    return result


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
        st.session_state.sector          = make_mock_sector(n=10, seed=42)
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

    st.session_state.sx_reco          = sx_reco
    st.session_state.certificate      = cert
    st.session_state.counterfactual   = None   # cleared; request via Explain button
    st.session_state.bl_action        = bl_action
    st.session_state.bl_score         = bl_score
    st.session_state.needs_recompute  = False


def _run_explainer():
    """Run the counterfactual explainer on demand (slow — ~30-60 s)."""
    sector  = st.session_state.sector
    sx_reco = st.session_state.sx_reco
    if sx_reco is None:
        return
    with st.spinner("Computing minimal-flip counterfactual…"):
        cf = st.session_state.explainer.explain(sector, sx_reco)
    st.session_state.counterfactual = cf


# ══════════════════════════════════════════════════════════════════════════════
# Sector mutation helpers  (each marks needs_recompute)
# ══════════════════════════════════════════════════════════════════════════════

def _load_new_sector(seed: int, n: int):
    from state_schema import make_mock_sector
    st.session_state.sector          = make_mock_sector(n=n, seed=seed)
    st.session_state.needs_recompute = True


def _step_60s():
    adapter = st.session_state.adapter
    sector  = st.session_state.sector
    adapter.load_scenario(sector)
    adapter.step(60.0)
    st.session_state.sector          = adapter.get_state()
    st.session_state.needs_recompute = True


def _inject_emergency(aircraft_id: str, emg_type: str):
    from state_schema import AircraftState, SectorState
    sector = st.session_state.sector

    def _patch(ac: AircraftState) -> AircraftState:
        if ac.id != aircraft_id:
            return ac
        return AircraftState(
            id=ac.id, lat=ac.lat, lon=ac.lon,
            altitude_ft=ac.altitude_ft, heading_deg=ac.heading_deg,
            ground_speed_kt=ac.ground_speed_kt,
            fuel_kg=ac.fuel_kg,
            fuel_burn_rate_kg_per_min=ac.fuel_burn_rate_kg_per_min,
            reserve_fuel_kg=ac.reserve_fuel_kg,
            emergency_flag=True,
            emergency_type=emg_type,
            runway_needed=ac.runway_needed or list(sector.runway_availability.keys())[0],
        )

    st.session_state.sector = SectorState(
        aircraft=[_patch(ac) for ac in sector.aircraft],
        runway_availability=dict(sector.runway_availability),
        sim_time_s=sector.sim_time_s,
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

    fig         = go.Figure()
    annotations = []                 # collected and added via update_layout
    label_pos   = _label_pos_map(sector.aircraft)

    # ── per-aircraft traces ────────────────────────────────────────────────
    for ac in sector.aircraft:
        is_emg  = ac.emergency_flag
        is_crit = ac.is_fuel_critical and not is_emg

        if is_emg:
            dot_col, dot_size, txt_col, txt_sz = _RED,    18, _RED,    12
        elif is_crit:
            dot_col, dot_size, txt_col, txt_sz = _ORANGE, 14, _ORANGE, 11
        else:
            frac    = min(1.0, ac.fuel_minutes_above_reserve / 120.0)
            dot_col = f"rgb({int(80*(1-frac)+20)},{int(180*frac+50)},80)"
            dot_size, txt_col, txt_sz = 9, '#c9d1d9', 10

        # 5-min projected path (dashed, faint)
        p_lat, p_lon = _project_position(
            ac.lat, ac.lon, ac.heading_deg, ac.ground_speed_kt, 5.0
        )
        fig.add_trace(go.Scatter(
            x=[ac.lon, p_lon], y=[ac.lat, p_lat],
            mode='lines',
            line=dict(color=dot_col, width=1.5, dash='dot'),
            opacity=0.35, showlegend=False, hoverinfo='skip',
        ))

        # Emergency pulsing halo
        if is_emg:
            fig.add_trace(go.Scatter(
                x=[ac.lon], y=[ac.lat], mode='markers',
                marker=dict(size=54, color='rgba(248,81,73,0.10)',
                            line=dict(color='rgba(248,81,73,0.50)', width=1.5)),
                showlegend=False, hoverinfo='skip',
            ))

        # Heading arrow
        hdg_rad = math.radians(ac.heading_deg)
        fig.add_trace(go.Scatter(
            x=[ac.lon, ac.lon + _ARROW_LEN * math.sin(hdg_rad)],
            y=[ac.lat, ac.lat + _ARROW_LEN * math.cos(hdg_rad)],
            mode='lines', line=dict(color=dot_col, width=2),
            showlegend=False, hoverinfo='skip',
        ))

        # Main dot + label
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
            x=[ac.lon], y=[ac.lat],
            mode='markers+text',
            marker=dict(size=dot_size, color=dot_col,
                        line=dict(color='white', width=2.0 if is_emg else 1.2)),
            text=[ac.id], textposition=label_pos[ac.id],
            textfont=dict(color=txt_col, size=txt_sz),
            name=ac.id, hovertemplate=hover, showlegend=False,
        ))

    # ── airport markers + two-line annotations ────────────────────────────
    for icao, ap in _AIRPORTS.items():
        rwy_str = ', '.join(ap['runways']) if ap['runways'] else '—'
        fig.add_trace(go.Scatter(
            x=[ap['lon']], y=[ap['lat']], mode='markers',
            marker=dict(size=16, symbol='square', color='#0a1628',
                        line=dict(color=_BLUE, width=2.5)),
            showlegend=False,
            hovertemplate=(f"<b>{icao}</b>  {ap['city']}, {ap['country']}<br>"
                           f"Runways: {rwy_str}<extra></extra>"),
        ))
        annotations.append(dict(
            x=ap['lon'], y=ap['lat'],
            text=(f"<b>{icao}</b><br>"
                  f"<span style='font-size:9px;color:#8b949e'>"
                  f"{ap['city']} · {ap['country']}</span>"),
            showarrow=False, yshift=26, xshift=2,
            bgcolor='rgba(13,17,23,0.88)',
            bordercolor=_BLUE, borderwidth=1, borderpad=4,
            font=dict(color=_BLUE, size=11), align='center',
        ))

    # ── action overlay ────────────────────────────────────────────────────
    animated_trace_idx = None
    anim_start = anim_end = (0.0, 0.0)

    if isinstance(active_action, AssignRunwayAction):
        tgt_ac  = next((a for a in sector.aircraft
                        if a.id == active_action.aircraft_id), None)
        ap_icao = _RUNWAY_AIRPORT.get(active_action.runway_id)
        tgt_ap  = _AIRPORTS.get(ap_icao) if ap_icao else None

        if tgt_ac and tgt_ap:
            # Ghost: faded marker at the CURRENT position (the "before")
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon], y=[tgt_ac.lat], mode='markers',
                marker=dict(size=18, color='rgba(248,81,73,0.25)',
                            line=dict(color='rgba(248,81,73,0.55)', width=1.5)),
                showlegend=False, hoverinfo='skip',
            ))
            # Amber reroute line — where SKYLANCE-X is SENDING the aircraft
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon, tgt_ap['lon']],
                y=[tgt_ac.lat, tgt_ap['lat']],
                mode='lines', line=dict(color=_ORANGE, width=2.5),
                showlegend=False, hoverinfo='skip',
            ))
            # Animated dot (updated by frames; added last — index is known)
            anim_start = (tgt_ac.lat, tgt_ac.lon)
            anim_end   = (tgt_ap['lat'], tgt_ap['lon'])
            animated_trace_idx = len(fig.data)
            fig.add_trace(go.Scatter(
                x=[tgt_ac.lon], y=[tgt_ac.lat], mode='markers',
                marker=dict(size=18, color=_ORANGE,
                            line=dict(color='white', width=2.5)),
                showlegend=False,
                hovertemplate=(f"<b>{tgt_ac.id}</b> → {ap_icao}<br>"
                               f"Runway {active_action.runway_id}"
                               f"<extra></extra>"),
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

    # ── legend annotation ─────────────────────────────────────────────────
    path_legend = (
        "   <span style='color:#e3a135'>━━</span> rerouted path"
        if has_anim else ""
    )
    annotations.append(dict(
        text=(
            "⬤ <span style='color:#3fb950'>normal</span>"
            "  ⬤ <span style='color:#e3a135'>critical</span>"
            "  ⬤ <span style='color:#f85149'>emergency</span>"
            "  ⬛ airport<br>"
            "<span style='color:#444'>┄┄</span> 5-min path" + path_legend
        ),
        xref='paper', yref='paper', x=0.01, y=0.01,
        xanchor='left', yanchor='bottom', showarrow=False,
        bgcolor='rgba(13,17,23,0.82)', bordercolor='#30363d',
        borderwidth=1, borderpad=5,
        font=dict(color=_TEXT, size=9), align='left',
    ))

    # Scale note
    annotations.append(dict(
        text='1° lat ≈ 60 nm',
        xref='paper', yref='paper', x=0.99, y=0.01,
        xanchor='right', yanchor='bottom', showarrow=False,
        font=dict(color='#484f58', size=9),
    ))

    # ── layout ────────────────────────────────────────────────────────────
    all_lats = ([ac.lat for ac in sector.aircraft]
                + [ap['lat'] for ap in _AIRPORTS.values()])
    all_lons = ([ac.lon for ac in sector.aircraft]
                + [ap['lon'] for ap in _AIRPORTS.values()])

    fig.update_layout(
        showlegend=False,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        height=490,
        margin=dict(l=10, r=10, t=30, b=55 if has_anim else 10),
        title=dict(
            text=(f"Sector radar  ·  t = {sector.sim_time_s:.0f} s  "
                  f"·  {len(sector.aircraft)} aircraft"),
            font=dict(color=_TEXT, size=13), x=0.02,
        ),
        xaxis=dict(
            range=[min(all_lons) - 1.0, max(all_lons) + 1.0],
            gridcolor=_GRID, tickfont=dict(color=_GREEN),
            zerolinecolor=_GRID,
            title=dict(text='Longitude °E', font=dict(color=_GREEN)),
        ),
        yaxis=dict(
            range=[min(all_lats) - 0.6, max(all_lats) + 0.6],
            gridcolor=_GRID, tickfont=dict(color=_GREEN),
            zerolinecolor=_GRID,
            title=dict(text='Latitude °N', font=dict(color=_GREEN)),
            scaleanchor='x', scaleratio=1.6,
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
            _run_explainer()

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
