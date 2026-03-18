"""
FreeGSNKE ↔ TORAX (or simplified) coupling loop.

Implements the self-consistent loop described in docs/physics_codes.md §4:
  1. FreeGSNKE step: solve GS + circuit equations → psi(R,Z), q(rho), Ip
  2. Write GEQDSK → TORAX reads geometry (or in-memory for simplified backend)
  3. Transport step: evolve Te, ne, j, Ti
  4. Extract betap, Ip from transport → feed back to FreeGSNKE profiles
  5. Repeat

The coupling is "loose" (file-based GEQDSK exchange) for maximum compatibility.
For the simplified backend, geometry is passed in-memory (no files needed).
"""

from __future__ import annotations

import os
import tempfile
import time as _time
import warnings
from typing import Callable

import numpy as np

from .equilibrium import EquilibriumSolver
from .transport import SimplifiedTransport, TransportSolver, TransportState


class CoupledSimulator:
    """
    Coupled FreeGSNKE + transport simulator.

    Parameters
    ----------
    eq_solver  : EquilibriumSolver
    tr_solver  : TransportSolver
    iter_params: dict with ITER machine parameters (R_major, a_minor, B0, ...)
    dt_couple  : float, coupling time step (s). Default 1.0 s.
    verbose    : bool, print progress.
    """

    def __init__(
        self,
        eq_solver: EquilibriumSolver,
        tr_solver: TransportSolver,
        iter_params: dict | None = None,
        dt_couple: float = 1.0,
        verbose: bool = True,
    ):
        self.eq  = eq_solver
        self.tr  = tr_solver
        self.dt  = dt_couple
        self.verbose = verbose

        if iter_params is None:
            from .iter_machine import ITER_PARAMS
            iter_params = ITER_PARAMS
        self.iter_params = iter_params

        self._geqdsk_dir = tempfile.mkdtemp(prefix="iter_geqdsk_")
        self._step_count = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(
        self,
        Ip_A: float = 15.0e6,
        betap: float = 0.5,
        T_e0_keV: float = 20.0,
        n_e0_m3: float = 1.0e20,
        coil_currents: dict | None = None,
        xpoints: list | None = None,
    ) -> tuple[dict, TransportState]:
        """
        Compute initial equilibrium and transport state.

        Returns
        -------
        eq_signals : dict (from EquilibriumSolver.get_signals())
        tr_state   : TransportState
        """
        if self.verbose:
            print(f"[Coupling] Computing initial equilibrium (Ip={Ip_A*1e-6:.1f} MA)...")

        # --- Static equilibrium ---
        eq = self.eq.solve_static(
            Ip=Ip_A,
            betap=betap,
            coil_targets=coil_currents,
            xpoints=xpoints,
        )
        eq_signals = self.eq.get_signals(eq)

        if self.verbose:
            print(f"  q95={eq_signals['q95']:.2f}  betaN={eq_signals['betaN']:.3f}"
                  f"  Ip={eq_signals['Ip']*1e-6:.2f} MA")

        # --- Write initial GEQDSK (for TORAX) ---
        geqdsk_path = None
        if self.tr.backend == "torax":
            geqdsk_path = os.path.join(self._geqdsk_dir, "eq_init.geqdsk")
            self.eq.write_geqdsk(geqdsk_path, eq)

        # --- Initial transport state ---
        tr_state = self.tr.init(
            geometry=eq_signals,
            Ip_A=Ip_A,
            T_e0_keV=T_e0_keV,
            n_e0_m3=n_e0_m3,
            geqdsk_path=geqdsk_path,
        )

        # --- Initialise dynamic equilibrium solver ---
        self.eq.init_dynamic(eq, betap=betap, Ip=Ip_A)

        self._last_eq_signals = eq_signals
        self._step_count = 0
        return eq_signals, tr_state

    # ------------------------------------------------------------------
    # Single coupling step
    # ------------------------------------------------------------------

    def step(
        self,
        tr_state: TransportState,
        V_coils: dict,
        sources: dict,
    ) -> tuple[dict, TransportState]:
        """
        Advance one coupling time step.

        Parameters
        ----------
        tr_state  : current TransportState
        V_coils   : dict {coil_name: voltage_V} for FreeGSNKE
        sources   : dict {P_ohm_W, P_aux_W, n_target, ...} for transport

        Returns
        -------
        eq_signals : updated equilibrium signals dict
        new_tr_state : updated TransportState
        """
        # 1. Extract profile parameters from transport → equilibrium input
        profile_params = self.tr.extract_freegsnke_profiles(tr_state, self._last_eq_signals or {})
        betap_new = profile_params["betap"]

        # 2. FreeGSNKE step (Ip from equilibrium is authoritative — circuit equations determine it)
        eq = self.eq.step(
            dt=self.dt,
            V_coils=V_coils,
            betap=betap_new,
            Ip=profile_params["Ip"],
        )
        eq_signals = self.eq.get_signals(eq)
        self._last_eq_signals = eq_signals

        # 3. Write GEQDSK for TORAX (if using TORAX backend)
        geqdsk_path = None
        if self.tr.backend == "torax":
            step_name  = f"eq_{self._step_count:05d}.geqdsk"
            geqdsk_path = os.path.join(self._geqdsk_dir, step_name)
            self.eq.write_geqdsk(geqdsk_path, eq)

        # 4. Transport step
        new_tr_state = self.tr.step(
            geometry=eq_signals,
            sources=sources,
            dt=self.dt,
            state=tr_state,
            geqdsk_path=geqdsk_path,
        )

        self._step_count += 1
        return eq_signals, new_tr_state

    # ------------------------------------------------------------------
    # Run a full trajectory
    # ------------------------------------------------------------------

    def run(
        self,
        t_end: float,
        Ip_waveform: Callable[[float], float],
        P_heat_waveform: Callable[[float], float],
        n_target_waveform: Callable[[float], float] | None = None,
        V_coils_waveform: Callable[[float], dict] | None = None,
        stop_condition: Callable[[dict, TransportState], bool] | None = None,
        record_every: int = 1,
    ) -> dict:
        """
        Run the coupled simulation from t=0 to t=t_end.

        Parameters
        ----------
        t_end              : float, end time (s)
        Ip_waveform        : callable(t) → Ip (A)
        P_heat_waveform    : callable(t) → P_heat (W)
        n_target_waveform  : callable(t) → n_e_target (m^-3), or None
        V_coils_waveform   : callable(t) → {coil_name: V}, or None
        stop_condition     : callable(eq_signals, tr_state) → bool
                             Return True to stop (disruption triggered)
        record_every       : int, record signals every N steps

        Returns
        -------
        trajectory : dict with time-series arrays:
          "time"     : [N_t]       float
          "Ip"       : [N_t]       A
          "betaN"    : [N_t]
          "q95"      : [N_t]
          "f_GW"     : [N_t]       Greenwald fraction
          "T_e"      : [N_rho, N_t] keV
          "n_e"      : [N_rho, N_t] m^-3
          "j_tor"    : [N_rho, N_t] A/m^2
          "q_profile": [N_rho, N_t]
          "W_thermal": [N_t]       J
          "disruption_time" : float or None
          "stopped"  : bool
        """
        n_steps  = int(t_end / self.dt)
        rho      = np.linspace(0, 1, self.tr.n_rho)

        # Pre-allocate storage
        record_times  = []
        Ip_arr        = []
        betaN_arr     = []
        q95_arr       = []
        f_GW_arr      = []
        W_arr         = []
        T_e_list      = []
        n_e_list      = []
        j_list        = []
        q_list        = []

        # Initial state
        Ip0    = Ip_waveform(0.0)
        P0     = P_heat_waveform(0.0)
        n0     = n_target_waveform(0.0) if n_target_waveform else None
        eq_sig, tr_state = self.init(Ip_A=Ip0, T_e0_keV=20.0, n_e0_m3=n0 or 1.0e20)

        # Record t=0
        self._record(eq_sig, tr_state, record_times, Ip_arr, betaN_arr, q95_arr,
                     f_GW_arr, W_arr, T_e_list, n_e_list, j_list, q_list)

        disruption_time = None
        stopped         = False

        t_start_wall = _time.time()

        for step_idx in range(n_steps):
            t = (step_idx + 1) * self.dt

            # Build inputs for this step
            V_coils = V_coils_waveform(t) if V_coils_waveform else {}
            sources = {
                "P_aux_W":  P_heat_waveform(t),
                "P_ohm_W":  0.0,   # ohmic power from equilibrium (not modelled here)
                "n_target": n_target_waveform(t) if n_target_waveform else None,
            }

            try:
                eq_sig, tr_state = self.step(tr_state, V_coils, sources)
            except Exception as exc:
                warnings.warn(f"Coupling step failed at t={t:.1f}s: {exc}")
                stopped = True
                break

            # Record
            if step_idx % record_every == 0:
                self._record(eq_sig, tr_state, record_times, Ip_arr, betaN_arr, q95_arr,
                             f_GW_arr, W_arr, T_e_list, n_e_list, j_list, q_list)

            # Check stop condition
            if stop_condition is not None and stop_condition(eq_sig, tr_state):
                disruption_time = t
                stopped = True
                if self.verbose:
                    print(f"[Coupling] Stop condition met at t={t:.2f}s")
                break

            if self.verbose and step_idx % 10 == 0:
                elapsed = _time.time() - t_start_wall
                print(f"  t={t:.1f}s  Ip={eq_sig['Ip']*1e-6:.2f}MA"
                      f"  q95={eq_sig['q95']:.2f}  betaN={eq_sig['betaN']:.3f}"
                      f"  Te0={float(tr_state.T_e[0]):.1f}keV"
                      f"  [{elapsed:.0f}s elapsed]")

        T_e_arr = np.column_stack(T_e_list) if T_e_list else np.zeros((self.tr.n_rho, 1))
        n_e_arr = np.column_stack(n_e_list) if n_e_list else np.zeros((self.tr.n_rho, 1))
        j_arr   = np.column_stack(j_list)   if j_list   else np.zeros((self.tr.n_rho, 1))
        q_arr   = np.column_stack(q_list)   if q_list   else np.zeros((self.tr.n_rho, 1))

        return {
            "time":            np.array(record_times),
            "Ip":              np.array(Ip_arr),
            "betaN":           np.array(betaN_arr),
            "q95":             np.array(q95_arr),
            "f_GW":            np.array(f_GW_arr),
            "W_thermal":       np.array(W_arr),
            "T_e":             T_e_arr,       # [n_rho, n_t]
            "n_e":             n_e_arr,
            "j_tor":           j_arr,
            "q_profile":       q_arr,
            "rho":             rho,
            "disruption_time": disruption_time,
            "stopped":         stopped,
        }

    def _record(self, eq_sig, tr_state, times, Ip_arr, betaN_arr, q95_arr,
                f_GW_arr, W_arr, T_e_list, n_e_list, j_list, q_list):
        """Append one time point to the recording lists."""
        from .iter_machine import ITER_PARAMS
        times.append(tr_state.time)
        Ip_arr.append(eq_sig.get("Ip",    np.nan))
        betaN_arr.append(eq_sig.get("betaN", np.nan))
        q95_arr.append(eq_sig.get("q95",  np.nan))
        W_arr.append(tr_state.W_thermal)
        T_e_list.append(tr_state.T_e.copy())
        n_e_list.append(tr_state.n_e.copy())
        j_list.append(tr_state.j_tor.copy())
        q_list.append(tr_state.q.copy())

        # Greenwald fraction — use actual a_minor from equilibrium if available
        Ip_MA = eq_sig.get("Ip", 15e6) * 1e-6
        a     = eq_sig.get("a_minor", ITER_PARAMS["a_minor"])
        f_GW  = tr_state.greenwald_fraction(Ip_MA, a)
        f_GW_arr.append(f_GW)

    def run_with_torax(
        self,
        t_end: float,
        Ip_waveform: Callable[[float], float],
        P_heat_waveform: Callable[[float], float],
        n_target_waveform: Callable[[float], float] | None = None,
        n_eq_steps: int = 10,
        transport_model: str = "constant",
    ) -> dict:
        """
        Run coupled simulation using TORAX for transport (full-run mode).

        Strategy:
        1. Pre-compute N equilibria at evenly spaced time points using
           FreeGSNKE + SimplifiedTransport for betap feedback
        2. Write GEQDSK files for each equilibrium
        3. Run TORAX once with time-dependent geometry
        4. Return trajectory with TORAX profiles

        Parameters
        ----------
        t_end              : float, end time (s)
        Ip_waveform        : callable(t) → Ip (A)
        P_heat_waveform    : callable(t) → P_heat (W)
        n_target_waveform  : callable(t) → n_e_target (m^-3), or None
        n_eq_steps         : int, number of equilibrium time points
        transport_model    : str, TORAX transport model ('constant', 'qlknn', etc.)

        Returns
        -------
        trajectory : dict with time-series arrays (same format as run())
        """
        if self.tr.backend != "torax":
            raise RuntimeError("run_with_torax() requires the TORAX transport backend.")

        if self.verbose:
            print(f"[Coupling] TORAX full-run: pre-computing {n_eq_steps} equilibria...")

        # Phase 1: Pre-compute equilibria using simplified transport for betap
        simple_tr = SimplifiedTransport(n_rho=self.tr.n_rho)
        dt_eq = t_end / n_eq_steps
        geometry_files = {}

        # Initial equilibrium
        Ip0 = Ip_waveform(0.0)
        eq = self.eq.solve_static(Ip=Ip0, betap=0.5, xpoints=[(6.0, -3.8)])
        eq_signals = self.eq.get_signals(eq)

        n0 = n_target_waveform(0.0) if n_target_waveform else 1.0e20
        tr_state = simple_tr.init(Ip0, T_e0_keV=20.0, n_e0_m3=n0, geometry=eq_signals)

        # Write initial GEQDSK
        fname = "eq_t0000.geqdsk"
        self.eq.write_geqdsk(os.path.join(self._geqdsk_dir, fname), eq)
        geometry_files[0.0] = fname

        # Initialise dynamic solver
        self.eq.init_dynamic(eq, betap=0.5, Ip=Ip0)

        eq_signals_list = [eq_signals]

        for i in range(1, n_eq_steps + 1):
            t = i * dt_eq

            # Simple transport step for betap feedback
            sources = {
                "P_aux_W": P_heat_waveform(t),
                "P_ohm_W": 0.0,
                "n_target": n_target_waveform(t) if n_target_waveform else None,
            }
            tr_state = simple_tr.step(tr_state, eq_signals, sources, dt_eq)

            # Extract betap for equilibrium feedback
            profile_params = self.tr.extract_freegsnke_profiles(tr_state, eq_signals)
            betap_new = profile_params["betap"]

            # Equilibrium step
            try:
                eq = self.eq.step(dt=dt_eq, V_coils={}, betap=betap_new, Ip=Ip_waveform(t))
                eq_signals = self.eq.get_signals(eq)
            except Exception as exc:
                if self.verbose:
                    print(f"  Equilibrium step failed at t={t:.1f}s: {exc}")
                break

            # Write GEQDSK
            fname = f"eq_t{i:04d}.geqdsk"
            self.eq.write_geqdsk(os.path.join(self._geqdsk_dir, fname), eq)
            geometry_files[t] = fname
            eq_signals_list.append(eq_signals)

            if self.verbose and i % max(1, n_eq_steps // 5) == 0:
                print(f"  t={t:.1f}s  Ip={eq_signals['Ip']*1e-6:.2f}MA"
                      f"  q95={eq_signals['q95']:.2f}  betap={betap_new:.3f}")

        if self.verbose:
            print(f"  Pre-computed {len(geometry_files)} equilibria. Running TORAX...")

        # Phase 2: Run TORAX with all geometry files
        Ip_flat = Ip_waveform(t_end / 2)  # representative Ip
        P_heat = P_heat_waveform(t_end / 2)
        torax_states = self.tr.run_trajectory(
            geometry_dir=self._geqdsk_dir,
            geometry_files=geometry_files,
            t_final=t_end,
            Ip_A=Ip_flat,
            P_heat_W=P_heat,
            transport_model=transport_model,
        )

        if self.verbose:
            print(f"  TORAX complete: {len(torax_states)} time steps")

        # Phase 3: Build trajectory dict from TORAX states + equilibrium signals
        rho = np.linspace(0, 1, self.tr.n_rho)
        record_times = [s.time for s in torax_states]
        T_e_list = [s.T_e for s in torax_states]
        n_e_list = [s.n_e for s in torax_states]
        j_list = [s.j_tor for s in torax_states]
        q_list = [s.q for s in torax_states]
        W_arr = [s.W_thermal for s in torax_states]

        # Interpolate equilibrium signals to TORAX time points
        eq_times = sorted(geometry_files.keys())
        eq_Ip = [es.get("Ip", np.nan) for es in eq_signals_list]
        eq_betaN = [es.get("betaN", np.nan) for es in eq_signals_list]
        eq_q95 = [es.get("q95", np.nan) for es in eq_signals_list]

        Ip_arr = np.interp(record_times, eq_times[:len(eq_Ip)], eq_Ip)
        betaN_arr = np.interp(record_times, eq_times[:len(eq_betaN)], eq_betaN)
        q95_arr = np.interp(record_times, eq_times[:len(eq_q95)], eq_q95)

        # Greenwald fraction from TORAX density + equilibrium Ip
        from .iter_machine import ITER_PARAMS
        f_GW_arr = []
        for i, state in enumerate(torax_states):
            Ip_MA = Ip_arr[i] * 1e-6
            a = ITER_PARAMS["a_minor"]
            f_GW_arr.append(state.greenwald_fraction(Ip_MA, a))

        return {
            "time":            np.array(record_times),
            "Ip":              np.array(Ip_arr),
            "betaN":           np.array(betaN_arr),
            "q95":             np.array(q95_arr),
            "f_GW":            np.array(f_GW_arr),
            "W_thermal":       np.array(W_arr),
            "T_e":             np.column_stack(T_e_list) if T_e_list else np.zeros((self.tr.n_rho, 1)),
            "n_e":             np.column_stack(n_e_list) if n_e_list else np.zeros((self.tr.n_rho, 1)),
            "j_tor":           np.column_stack(j_list) if j_list else np.zeros((self.tr.n_rho, 1)),
            "q_profile":       np.column_stack(q_list) if q_list else np.zeros((self.tr.n_rho, 1)),
            "rho":             rho,
            "disruption_time": None,
            "stopped":         False,
        }

    def cleanup(self):
        """Remove temporary GEQDSK files."""
        import shutil
        if os.path.isdir(self._geqdsk_dir):
            shutil.rmtree(self._geqdsk_dir, ignore_errors=True)
