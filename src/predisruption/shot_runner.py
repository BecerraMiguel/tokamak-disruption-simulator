"""
Single shot runner for the pre-disruption phase.

Generates one complete shot (normal or disruptive) using the coupled
FreeGSNKE + transport simulator. Returns a time-series dict ready to
be handed to the HDF5 writer.

Shot types:
  - Normal (label=0): Ip follows reference waveform, stays within limits,
    terminates gracefully with Ip ramp-down.
  - Disruptive (label=1): starts from reference, then a perturbation is
    injected (density puff, heating transient, etc.) that drives the plasma
    beyond a disruption trigger. Terminates at trigger crossing and hands
    off to DREAM.

Disruption triggers (from configs/generation.yaml):
  f_GW  > 0.95   (Greenwald density limit)
  betaN > 3.2    (Troyon beta limit)
  q95   < 2.2    (kink stability limit)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .coupling import CoupledSimulator
from .equilibrium import EquilibriumSolver
from .transport import TransportSolver, TransportState


# ---------------------------------------------------------------------------
# Trigger thresholds
# ---------------------------------------------------------------------------

TRIGGERS = {
    "f_GW":  0.95,   # Greenwald fraction limit
    "betaN": 3.2,    # Troyon beta-N limit
    "q95":   2.2,    # q95 lower bound
}


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Parameters for one shot.

    All waveforms are specified as (t, value) arrays; the runner interpolates.
    """
    # Plasma current waveform (A vs s)
    Ip_t:   np.ndarray = field(default_factory=lambda: np.array([0.0, 10.0, 80.0, 90.0]))
    Ip_val: np.ndarray = field(default_factory=lambda: np.array([0.0, 15e6, 15e6, 0.0]))

    # Auxiliary heating (W vs s)
    P_heat_t:   np.ndarray = field(default_factory=lambda: np.array([0.0, 10.0, 80.0]))
    P_heat_val: np.ndarray = field(default_factory=lambda: np.array([0.0, 33e6, 33e6]))

    # Target line-averaged density (m^-3 vs s)
    ne_t:   np.ndarray = field(default_factory=lambda: np.array([0.0, 10.0, 80.0]))
    ne_val: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0e20, 1.0e20]))

    # Perturbation for disruptive shots
    perturbation_type: Literal["density", "beta", "q95", "none"] = "none"
    perturbation_start: float = 50.0   # s, when to inject perturbation
    perturbation_amp:   float = 0.0    # amplitude (relative)

    # Shot end time
    t_end: float = 90.0   # s

    # Coupling time step
    dt: float = 1.0   # s


def reference_15MA_scenario(t_end: float = 90.0, dt: float = 1.0) -> ScenarioConfig:
    """Standard ITER 15 MA DT flat-top scenario."""
    return ScenarioConfig(
        Ip_t   = np.array([0.0, 12.0, 80.0, 90.0]),
        Ip_val = np.array([3e6, 15e6, 15e6,  3e6]),
        P_heat_t   = np.array([0.0,  12.0,  80.0]),
        P_heat_val = np.array([5e6,  33e6,  33e6]),
        ne_t   = np.array([0.0,   12.0,   80.0]),
        ne_val = np.array([3e19, 1.0e20, 1.0e20]),
        perturbation_type  = "none",
        perturbation_start = 50.0,
        perturbation_amp   = 0.0,
        t_end = t_end,
        dt    = dt,
    )


def disruptive_scenario(
    base: ScenarioConfig | None = None,
    perturbation_type: Literal["density", "beta", "q95"] = "density",
    perturbation_start: float = 50.0,
    perturbation_amp: float = 0.3,
    t_end: float = 75.0,
    dt: float = 1.0,
) -> ScenarioConfig:
    """
    Create a disruptive scenario by perturbing a base normal scenario.

    Parameters
    ----------
    perturbation_type : "density" (ramp ne → f_GW > 0.95)
                        "beta"    (boost heating → betaN > 3.2)
                        "q95"     (reduce Ip → q95 < 2.2)
    perturbation_start : float, time when perturbation begins (s)
    perturbation_amp   : float, relative amplitude (0.3 = 30% increase)
    """
    if base is None:
        base = reference_15MA_scenario(t_end=t_end, dt=dt)

    cfg = ScenarioConfig(
        Ip_t   = base.Ip_t.copy(),
        Ip_val = base.Ip_val.copy(),
        P_heat_t   = base.P_heat_t.copy(),
        P_heat_val = base.P_heat_val.copy(),
        ne_t   = base.ne_t.copy(),
        ne_val = base.ne_val.copy(),
        perturbation_type  = perturbation_type,
        perturbation_start = perturbation_start,
        perturbation_amp   = perturbation_amp,
        t_end = t_end,
        dt    = dt,
    )

    if perturbation_type == "density":
        # Ramp density by (1 + amp) after perturbation_start
        cfg.ne_t   = np.array([0.0, 12.0, perturbation_start, t_end])
        cfg.ne_val = np.array([3e19, 1.0e20, 1.0e20, 1.0e20 * (1.0 + perturbation_amp)])

    elif perturbation_type == "beta":
        # Boost auxiliary heating after perturbation_start
        cfg.P_heat_t   = np.array([0.0, 12.0, perturbation_start, t_end])
        cfg.P_heat_val = np.array([5e6, 33e6, 33e6, 33e6 * (1.0 + perturbation_amp)])

    elif perturbation_type == "q95":
        # Ramp Ip upward after perturbation_start to lower q95 (q95 ∝ 1/Ip)
        cfg.Ip_t   = np.array([0.0, 12.0, perturbation_start, t_end])
        cfg.Ip_val = np.array([3e6, 15e6, 15e6, 15e6 * (1.0 + perturbation_amp)])

    return cfg


# ---------------------------------------------------------------------------
# Shot runner
# ---------------------------------------------------------------------------

class ShotRunner:
    """
    Runs a single pre-disruption shot and returns a time-series dict.

    Parameters
    ----------
    eq_solver  : EquilibriumSolver
    tr_solver  : TransportSolver
    triggers   : dict of disruption trigger thresholds (defaults: TRIGGERS)
    verbose    : bool
    """

    def __init__(
        self,
        eq_solver: EquilibriumSolver,
        tr_solver: TransportSolver,
        triggers: dict | None = None,
        verbose: bool = True,
    ):
        self.eq  = eq_solver
        self.tr  = tr_solver
        self.triggers = triggers or TRIGGERS
        self.verbose  = verbose

    def run_shot(
        self,
        scenario: ScenarioConfig,
        shot_id: int = 0,
    ) -> dict:
        """
        Execute one shot.

        Parameters
        ----------
        scenario : ScenarioConfig
        shot_id  : int, identifier stored in output

        Returns
        -------
        result : dict with keys:
          "shot_id"           : int
          "label"             : 0 (normal) or 1 (disruptive)
          "disruption_time"   : float (s), NaN if normal
          "trigger"           : str, which limit was crossed (or "none")
          "time"              : [N_t] float array (s)
          "rho"               : [N_rho] float array
          "Ip"                : [N_t] A
          "betaN"             : [N_t]
          "q95"               : [N_t]
          "f_GW"              : [N_t]
          "W_thermal"         : [N_t] J
          "T_e"               : [N_rho, N_t] keV
          "n_e"               : [N_rho, N_t] m^-3
          "j_tor"             : [N_rho, N_t] A/m^2
          "q_profile"         : [N_rho, N_t]
          "disruption_state"  : TransportState at trigger crossing (or None)
        """
        is_disruptive = (scenario.perturbation_type != "none")

        if self.verbose:
            ptype = scenario.perturbation_type
            label = "DISRUPTIVE" if is_disruptive else "NORMAL"
            print(f"\n[ShotRunner] Shot {shot_id} — {label}"
                  + (f" ({ptype} perturbation)" if is_disruptive else ""))

        # Build waveform callables via linear interpolation
        Ip_wave   = _make_waveform(scenario.Ip_t,      scenario.Ip_val)
        P_wave    = _make_waveform(scenario.P_heat_t,  scenario.P_heat_val)
        ne_wave   = _make_waveform(scenario.ne_t,      scenario.ne_val)

        # Disruption stop condition
        triggered_by = [None]   # mutable closure

        def stop_condition(eq_signals: dict, tr_state: TransportState) -> bool:
            q95   = eq_signals.get("q95",   99.0)
            betaN = eq_signals.get("betaN",  0.0)
            from .iter_machine import ITER_PARAMS
            Ip_MA = eq_signals.get("Ip", 15e6) * 1e-6
            f_GW  = tr_state.greenwald_fraction(Ip_MA, ITER_PARAMS["a_minor"])

            if f_GW  > self.triggers["f_GW"]:
                triggered_by[0] = "f_GW"
                return True
            if betaN > self.triggers["betaN"]:
                triggered_by[0] = "betaN"
                return True
            if q95   < self.triggers["q95"]:
                triggered_by[0] = "q95"
                return True
            return False

        # Create coupled simulator
        simulator = CoupledSimulator(
            eq_solver=self.eq,
            tr_solver=self.tr,
            dt_couple=scenario.dt,
            verbose=self.verbose,
        )

        # Run
        trajectory = simulator.run(
            t_end             = scenario.t_end,
            Ip_waveform       = Ip_wave,
            P_heat_waveform   = P_wave,
            n_target_waveform = ne_wave,
            stop_condition    = stop_condition if is_disruptive else None,
        )

        simulator.cleanup()

        # Determine label and trigger
        trigger     = triggered_by[0] or "none"
        disruption_time = trajectory.get("disruption_time")
        label = 1 if (disruption_time is not None) else 0

        if is_disruptive and label == 0:
            warnings.warn(
                f"Shot {shot_id}: expected disruption but no trigger was crossed. "
                "Consider increasing perturbation_amp."
            )

        if self.verbose:
            if label == 1:
                print(f"  → Disruption at t={disruption_time:.2f}s  trigger={trigger}")
            else:
                print(f"  → Normal shot, t_end={trajectory['time'][-1]:.1f}s")

        return {
            "shot_id":          shot_id,
            "label":            label,
            "disruption_time":  disruption_time if disruption_time is not None else float("nan"),
            "trigger":          trigger,
            "time":             trajectory["time"],
            "rho":              trajectory["rho"],
            "Ip":               trajectory["Ip"],
            "betaN":            trajectory["betaN"],
            "q95":              trajectory["q95"],
            "f_GW":             trajectory["f_GW"],
            "W_thermal":        trajectory["W_thermal"],
            "T_e":              trajectory["T_e"],
            "n_e":              trajectory["n_e"],
            "j_tor":            trajectory["j_tor"],
            "q_profile":        trajectory["q_profile"],
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_waveform(t_arr: np.ndarray, val_arr: np.ndarray):
    """Return a callable that linearly interpolates (t_arr, val_arr)."""
    def waveform(t: float) -> float:
        return float(np.interp(t, t_arr, val_arr))
    return waveform
