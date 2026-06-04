"""
BlueSky adapter for SKYLANCE-X.

Wraps BlueSky behind the state_schema interface so the rest of the system
never touches BlueSky directly.

Unit notes (BlueSky internal SI):
  altitude  : metres       → our schema: feet   (÷ ft = 0.3048)
  speed     : m/s          → our schema: knots  (÷ kts = 0.514444)
  fuelflow  : kg/s         → our schema: kg/min (× 60)

Fuel tracking: BlueSky's OpenAP mass field is initialised once and not
decremented. We maintain _fuel_kg[acid] ourselves, decrementing by
perf.fuelflow[i] * simdt each step.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

import bluesky as bs
from bluesky.core import simtime as _simtime
from bluesky.tools.aero import ft, kts

from state_schema import AircraftState, SectorState

# --------------------------------------------------------------------------
# Unit conversion constants
# --------------------------------------------------------------------------
_M2FT = 1.0 / ft          # metres → feet
_MS2KT = 1.0 / kts        # m/s → knots


class BlueSkyAdapter:
    """
    Presents a clean SKYLANCE-X interface over BlueSky.

    Only one BlueSky sim process can exist per Python interpreter (bs.* are
    module-level singletons), so instantiate this class once and reuse it.
    """

    # Coarser simulation timestep for the dashboard.  Default BlueSky dt is
    # 0.05 s (20 Hz); 0.5 s is 10× faster with negligible accuracy loss for
    # a 5-min lookahead.  Pass simdt=0.05 to restore full fidelity.
    def __init__(self, simdt: float = 0.5) -> None:
        self._initialized = False
        self._simdt = simdt
        self._debug_done = False   # one-shot debug dump flag

        # Fuel state: BlueSky's OpenAP mass is never decremented, so we
        # shadow it ourselves and drain it each sim step.
        self._fuel_kg: dict[str, float] = {}

        # Fields BlueSky has no model for — carried from the scenario.
        # Speed and burn rate are also shadowed because:
        #   - bs.traf.gs returns TAS (not GS) after BlueSky's CAS→TAS
        #     altitude conversion, making it systematically too high.
        #   - bs.traf.perf.fuelflow is 0 until the physics model runs a
        #     full step; we use our own value so the very first get_state()
        #     call returns a meaningful burn rate.
        self._ground_speed_kt: dict[str, float] = {}
        self._burn_rate_kg_per_min: dict[str, float] = {}
        self._reserve_fuel: dict[str, float] = {}
        self._emergency_flag: dict[str, bool] = {}
        self._emergency_type: dict[str, Optional[str]] = {}
        self._runway_needed: dict[str, Optional[str]] = {}
        self._runway_availability: dict[str, bool] = {}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_scenario(self, sector: SectorState) -> None:
        """
        Clear BlueSky and spawn the aircraft described by `sector`.

        Aircraft type defaults to A320 for all aircraft because AircraftState
        does not carry a type field. Override by subclassing if needed.
        """
        self._ensure_init()
        # Clear traffic without bs.sim.reset(), which breaks when called
        # repeatedly from outside BlueSky's normal entry point.
        bs.stack.stack('DEL ALL')
        # DEL ALL is queued asynchronously; drain any remaining aircraft
        # directly by index (delete(0) reindexes, so looping from front is safe).
        while bs.traf.ntraf > 0:
            bs.traf.delete(0)
        bs.stack.stack('CDMETHOD STATEBASED')

        self._fuel_kg.clear()
        self._ground_speed_kt.clear()
        self._burn_rate_kg_per_min.clear()
        self._reserve_fuel.clear()
        self._emergency_flag.clear()
        self._emergency_type.clear()
        self._runway_needed.clear()

        for ac in sector.aircraft:
            bs.traf.cre(
                acid=ac.id,
                actype='A320',
                aclat=ac.lat,
                aclon=ac.lon,
                achdg=ac.heading_deg,
                acalt=ac.altitude_ft * ft,    # schema ft → BlueSky metres
                acspd=ac.ground_speed_kt * kts,  # schema kts → BlueSky m/s (as CAS)
            )
            self._fuel_kg[ac.id]              = ac.fuel_kg
            self._ground_speed_kt[ac.id]      = ac.ground_speed_kt
            self._burn_rate_kg_per_min[ac.id] = ac.fuel_burn_rate_kg_per_min
            self._reserve_fuel[ac.id]         = ac.reserve_fuel_kg
            self._emergency_flag[ac.id]       = ac.emergency_flag
            self._emergency_type[ac.id]       = ac.emergency_type
            self._runway_needed[ac.id]        = ac.runway_needed

        self._runway_availability = dict(sector.runway_availability)

        # Ensure coarser timestep is always active (accuracy is fine for a
        # 5-minute lookahead: at 450 kt, 0.5 s error ≈ 0.06 nm — negligible).
        _simtime.setdt(self._simdt)

        # Align sim clock to scenario time before operating
        _simtime._clock.t = Decimal(repr(sector.sim_time_s))
        _simtime._clock.ft = sector.sim_time_s
        bs.sim.simt = sector.sim_time_s
        bs.sim.op()

    def get_state(self) -> SectorState:
        """Return a SectorState reflecting BlueSky's current state."""
        if not self._debug_done and bs.traf.ntraf > 0:
            print("DEBUG bs.traf attrs:", dir(bs.traf))
            print("DEBUG perf attrs:", dir(bs.traf.perf))
            print("DEBUG mass:", bs.traf.perf.mass[0] if hasattr(bs.traf.perf, 'mass') else 'NONE')
            print("DEBUG fuelflow:", bs.traf.perf.fuelflow[0] if hasattr(bs.traf.perf, 'fuelflow') else 'NONE')
            self._debug_done = True

        aircraft: list[AircraftState] = []
        for i in range(bs.traf.ntraf):
            acid = bs.traf.id[i]
            # Use shadowed values: bs.traf.gs returns TAS (BlueSky converts
            # our CAS input to TAS at altitude), and fuelflow is 0 until the
            # physics model has stepped at least once.
            aircraft.append(AircraftState(
                id=acid,
                lat=float(bs.traf.lat[i]),
                lon=float(bs.traf.lon[i]),
                altitude_ft=float(bs.traf.alt[i]) * _M2FT,
                heading_deg=float(bs.traf.hdg[i]),
                ground_speed_kt=self._ground_speed_kt.get(acid, 0.0),
                fuel_kg=self._fuel_kg.get(acid, 0.0),
                fuel_burn_rate_kg_per_min=self._burn_rate_kg_per_min.get(acid, 0.0),
                reserve_fuel_kg=self._reserve_fuel.get(acid, 0.0),
                emergency_flag=self._emergency_flag.get(acid, False),
                emergency_type=self._emergency_type.get(acid),
                runway_needed=self._runway_needed.get(acid),
            ))
        return SectorState(
            aircraft=aircraft,
            runway_availability=dict(self._runway_availability),
            sim_time_s=bs.sim.simt,
        )

    def step(self, seconds: float) -> None:
        """Advance the live simulation by `seconds` of sim time."""
        n = max(1, round(seconds / bs.sim.simdt))
        for _ in range(n):
            bs.sim.step()
        # Drain fuel once per step() call (same integral, much less Python overhead
        # than calling _drain_fuel() inside the tight 1200-iteration tick loop).
        self._drain_fuel(seconds)

    def fast_forward(self, seconds: float) -> SectorState:
        """
        Return the SectorState that would result from advancing by `seconds`,
        without permanently mutating the live simulation.

        The cascade suggestion engine calls this to probe future states.
        Implementation: snapshot all kinematic arrays → step → read → restore.
        """
        snap = self._snapshot()
        try:
            self.step(seconds)
            return self.get_state()
        finally:
            self._restore(snap)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if not self._initialized:
            bs.init(mode='sim', detached=True)
            bs.stack.stack('CDMETHOD STATEBASED')
            # navdb.reset() re-reads the navdata pickle on every sim.reset().
            # The navdb is purely static reference data (waypoints, airports) that
            # never changes between scenarios, so replace it with a no-op to avoid
            # a ~1-2s pickle reload on every load_scenario() call.
            bs.navdb.reset = lambda: None
            self._initialized = True

    def _drain_fuel(self, dt_s: float) -> None:
        """Burn fuel for each aircraft for dt_s seconds using the shadowed burn rate."""
        for acid in list(self._fuel_kg):
            rate = self._burn_rate_kg_per_min.get(acid, 0.0)  # kg/min
            burned_kg = rate / 60.0 * dt_s
            self._fuel_kg[acid] = max(0.0, self._fuel_kg[acid] - burned_kg)

    def _snapshot(self) -> dict:
        """
        Capture all state needed to restore BlueSky after a fast_forward call.

        Only kinematic + performance arrays are copied; the child Entity
        objects (trails, ADSB, etc.) are not needed for trajectory lookahead.
        """
        t = bs.traf
        return {
            # simulation clock (Decimal for lossless restore)
            'clock_t': _simtime._clock.t,
            'simt': bs.sim.simt,
            # aircraft identity
            'id': list(t.id),
            'type': list(t.type),
            # position / kinematics
            'lat':      t.lat.copy(),
            'lon':      t.lon.copy(),
            'alt':      t.alt.copy(),
            'hdg':      t.hdg.copy(),
            'trk':      t.trk.copy(),
            'tas':      t.tas.copy(),
            'cas':      t.cas.copy(),
            'gs':       t.gs.copy(),
            'gsnorth':  t.gsnorth.copy(),
            'gseast':   t.gseast.copy(),
            'vs':       t.vs.copy(),
            'M':        t.M.copy(),
            'ax':       t.ax.copy(),
            'coslat':   t.coslat.copy(),
            'distflown': t.distflown.copy(),
            # atmosphere
            'p':    t.p.copy(),
            'rho':  t.rho.copy(),
            'Temp': t.Temp.copy(),
            # autopilot commands
            'selspd':    t.selspd.copy(),
            'aptas':     t.aptas.copy(),
            'selalt':    t.selalt.copy(),
            'selvs':     t.selvs.copy(),
            'swlnav':    t.swlnav.copy(),
            'swvnav':    t.swvnav.copy(),
            'swvnavspd': t.swvnavspd.copy(),
            'swhdgsel':  t.swhdgsel.copy(),
            # energy tracking
            'work': t.work.copy(),
            # performance model
            'perf_fuelflow': t.perf.fuelflow.copy(),
            'perf_thrust':   t.perf.thrust.copy(),
            'perf_drag':     t.perf.drag.copy(),
            'perf_mass':     t.perf.mass.copy(),
            # our fuel shadow
            'fuel_kg': dict(self._fuel_kg),
        }

    def _restore(self, snap: dict) -> None:
        """Restore BlueSky to a previously snapshotted state."""
        t = bs.traf

        # Guard: if traffic count changed during fast_forward (shouldn't happen
        # in normal use), skip restore and let the caller deal with it.
        if t.ntraf != len(snap['lat']):
            raise RuntimeError(
                f"fast_forward changed ntraf from {len(snap['lat'])} to {t.ntraf}; "
                "restore aborted — aircraft were created or deleted mid-lookahead"
            )

        # Restore clock
        _simtime._clock.t = snap['clock_t']
        _simtime._clock.ft = float(snap['clock_t'])
        bs.sim.simt = snap['simt']

        # Restore identity
        t.id[:] = snap['id']
        t.type[:] = snap['type']

        # Restore kinematics
        t.lat[:]       = snap['lat']
        t.lon[:]       = snap['lon']
        t.alt[:]       = snap['alt']
        t.hdg[:]       = snap['hdg']
        t.trk[:]       = snap['trk']
        t.tas[:]       = snap['tas']
        t.cas[:]       = snap['cas']
        t.gs[:]        = snap['gs']
        t.gsnorth[:]   = snap['gsnorth']
        t.gseast[:]    = snap['gseast']
        t.vs[:]        = snap['vs']
        t.M[:]         = snap['M']
        t.ax[:]        = snap['ax']
        t.coslat[:]    = snap['coslat']
        t.distflown[:] = snap['distflown']

        # Restore atmosphere
        t.p[:]    = snap['p']
        t.rho[:]  = snap['rho']
        t.Temp[:] = snap['Temp']

        # Restore autopilot
        t.selspd[:]    = snap['selspd']
        t.aptas[:]     = snap['aptas']
        t.selalt[:]    = snap['selalt']
        t.selvs[:]     = snap['selvs']
        t.swlnav[:]    = snap['swlnav']
        t.swvnav[:]    = snap['swvnav']
        t.swvnavspd[:] = snap['swvnavspd']
        t.swhdgsel[:]  = snap['swhdgsel']

        # Restore energy / perf
        t.work[:]           = snap['work']
        t.perf.fuelflow[:]  = snap['perf_fuelflow']
        t.perf.thrust[:]    = snap['perf_thrust']
        t.perf.drag[:]      = snap['perf_drag']
        t.perf.mass[:]      = snap['perf_mass']

        # Restore our fuel shadow
        self._fuel_kg = dict(snap['fuel_kg'])


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from state_schema import make_mock_sector

    print("=== SKYLANCE-X BlueSky Adapter smoke-test ===\n")

    sector = make_mock_sector(n=12)
    adapter = BlueSkyAdapter()

    print(f"Loading {len(sector.aircraft)} aircraft into BlueSky...")
    adapter.load_scenario(sector)

    before = adapter.get_state()
    print(f"\n--- State at t={before.sim_time_s:.1f}s ---")
    for ac in before.aircraft:
        tag = " [EMERGENCY]" if ac.emergency_flag else ""
        print(
            f"  {ac.id:<8}  {ac.lat:7.3f}°N  {ac.lon:6.3f}°E  "
            f"FL{ac.altitude_ft/100:03.0f}  {ac.ground_speed_kt:.0f}kt  "
            f"fuel={ac.fuel_kg:.0f}kg  burn={ac.fuel_burn_rate_kg_per_min:.1f}kg/min"
            f"{tag}"
        )

    print("\nStepping 60 seconds...")
    adapter.step(60.0)

    after = adapter.get_state()
    print(f"\n--- State at t={after.sim_time_s:.1f}s ---")
    for b4, af in zip(before.aircraft, after.aircraft):
        dlat = af.lat - b4.lat
        dlon = af.lon - b4.lon
        dfuel = af.fuel_kg - b4.fuel_kg
        print(
            f"  {af.id:<8}  Δlat={dlat:+.4f}°  Δlon={dlon:+.4f}°  "
            f"Δfuel={dfuel:+.1f}kg  "
            f"burn_rate={af.fuel_burn_rate_kg_per_min:.1f}kg/min"
        )

    print("\nTesting fast_forward (60s lookahead, should not alter live state)...")
    live_before = adapter.get_state()
    lookahead = adapter.fast_forward(60.0)
    live_after = adapter.get_state()

    assert live_before.sim_time_s == live_after.sim_time_s, "fast_forward mutated sim_time!"
    for b4, af in zip(live_before.aircraft, live_after.aircraft):
        assert b4.lat == af.lat, f"fast_forward mutated lat of {b4.id}!"
        assert b4.fuel_kg == af.fuel_kg, f"fast_forward mutated fuel of {b4.id}!"

    print(f"  Lookahead sim_time: {lookahead.sim_time_s:.1f}s  "
          f"(live sim still at {live_after.sim_time_s:.1f}s) ✓")
    print(f"  Lookahead first aircraft: {lookahead.aircraft[0].id} "
          f"lat={lookahead.aircraft[0].lat:.4f}° "
          f"fuel={lookahead.aircraft[0].fuel_kg:.1f}kg")
    print("\nAll assertions passed.")
