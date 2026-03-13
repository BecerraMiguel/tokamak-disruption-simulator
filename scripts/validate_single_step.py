"""
Validate a single equilibrium + transport step.

Verifies that after bug fixes (Bug A: Te collapse, Bug B: W_thermal=0),
one equilibrium solve + one transport step produces physically reasonable
ITER-like results.

Run from project root:
    python scripts/validate_single_step.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    os.makedirs("results", exist_ok=True)

    results = {}  # {check_name: (passed, message)}

    # ------------------------------------------------------------------
    # 1. Build ITER machine
    # ------------------------------------------------------------------
    print("=" * 60)
    print("VALIDATION: Single equilibrium + transport step")
    print("=" * 60)

    print("\n[1] Building ITER machine...")
    from predisruption.iter_machine import build_iter_machine, ITER_PARAMS
    tokamak, active_coils, domain = build_iter_machine()

    # ------------------------------------------------------------------
    # 2. Create EquilibriumSolver and solve static equilibrium
    # ------------------------------------------------------------------
    print("\n[2] Solving static equilibrium (Ip=15 MA, betap=0.5)...")
    from predisruption.equilibrium import EquilibriumSolver

    eq_solver = EquilibriumSolver(
        tokamak,
        nx=65, ny=65,
        Rmin=domain[0], Rmax=domain[1],
        Zmin=domain[2], Zmax=domain[3],
    )

    try:
        eq = eq_solver.solve_static(Ip=15e6, betap=0.5)
        print("  Static equilibrium solved.")
    except Exception as exc:
        print(f"  ERROR: static solve failed: {exc}")
        _print_summary(results)
        return

    # ------------------------------------------------------------------
    # 3. Extract equilibrium signals
    # ------------------------------------------------------------------
    print("\n[3] Extracting equilibrium signals...")
    eq_signals = eq_solver.get_signals(eq)
    print(f"  R_axis = {eq_signals['R_axis']:.3f} m")
    print(f"  q95    = {eq_signals['q95']:.3f}")
    print(f"  betaN  = {eq_signals['betaN']:.4f}")
    print(f"  betap  = {eq_signals['betap']:.4f}")
    print(f"  kappa  = {eq_signals['kappa']:.3f}")
    print(f"  a_minor= {eq_signals['a_minor']:.3f} m")

    # ------------------------------------------------------------------
    # 4. Create transport solver and initialise
    # ------------------------------------------------------------------
    print("\n[4] Initialising simplified transport...")
    from predisruption.transport import TransportSolver

    tr_solver = TransportSolver(backend="simplified", n_rho=51)
    state = tr_solver.init(
        geometry=eq_signals,
        Ip_A=15e6,
        T_e0_keV=20.0,
        n_e0_m3=1e20,
    )

    print(f"  Te(0)      = {state.T_e[0]:.2f} keV")
    print(f"  ne(0)      = {state.n_e[0]:.3e} m^-3")
    print(f"  W_thermal  = {state.W_thermal:.3e} J")

    # Check Bug B: W_thermal > 0 after init
    w_init_ok = state.W_thermal > 0
    results["W_thermal > 0 after init (Bug B)"] = (
        w_init_ok,
        f"W_thermal = {state.W_thermal:.3e} J",
    )
    print(f"  W_thermal > 0 after init: {'PASS' if w_init_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # 5. Run one transport step
    # ------------------------------------------------------------------
    print("\n[5] Running one transport step (dt=1.0 s, P_aux=33 MW)...")
    sources = {"P_aux_W": 33e6, "P_ohm_W": 0.0}
    state = tr_solver.step(
        geometry=eq_signals,
        sources=sources,
        dt=1.0,
        state=state,
    )

    Te0 = float(state.T_e[0])
    ne0 = float(state.n_e[0])
    W = state.W_thermal
    print(f"  Te(0)      = {Te0:.2f} keV")
    print(f"  ne(0)      = {ne0:.3e} m^-3")
    print(f"  W_thermal  = {W:.3e} J")

    # Check Bug A: Te(0) stays between 5 and 50 keV
    te_ok = 5.0 < Te0 < 50.0
    results["5 < Te(0) < 50 keV (Bug A)"] = (te_ok, f"Te(0) = {Te0:.2f} keV")

    # ------------------------------------------------------------------
    # 6. Extract betap from transport and verify
    # ------------------------------------------------------------------
    print("\n[6] Extracting betap from transport profiles...")
    profile_params = tr_solver.extract_freegsnke_profiles(state, eq_signals)
    betap = profile_params["betap"]
    print(f"  betap (transport) = {betap:.4f}")

    betap_ok = 0.01 < betap < 5.0
    results["0.01 < betap < 5.0"] = (betap_ok, f"betap = {betap:.4f}")

    # ------------------------------------------------------------------
    # 7. Dynamic equilibrium step
    # ------------------------------------------------------------------
    print("\n[7] Initialising dynamic equilibrium and stepping...")
    try:
        eq_solver.init_dynamic(eq, betap=betap, Ip=15e6)
        eq_new = eq_solver.step(dt=1.0, V_coils={}, betap=betap, Ip=15e6)
        eq_signals_new = eq_solver.get_signals(eq_new)
        R_axis_new = eq_signals_new["R_axis"]
        print(f"  R_axis after step = {R_axis_new:.3f} m")
        raxis_ok = 5.0 < R_axis_new < 7.5
        results["R_axis ~6.2 m after dynamic step"] = (
            raxis_ok,
            f"R_axis = {R_axis_new:.3f} m",
        )
    except Exception as exc:
        print(f"  WARNING: dynamic step failed: {exc}")
        results["R_axis ~6.2 m after dynamic step"] = (False, f"Error: {exc}")

    # ------------------------------------------------------------------
    # Physics assertions
    # ------------------------------------------------------------------
    ne_ok = 1e19 < ne0 < 5e20
    results["1e19 < ne(0) < 5e20 m^-3"] = (ne_ok, f"ne(0) = {ne0:.3e} m^-3")

    w_ok = 100e3 < W < 1e9
    results["100 kJ < W_thermal < 1 GJ"] = (w_ok, f"W_thermal = {W:.3e} J")

    # ------------------------------------------------------------------
    # 8. Plot profiles (2x2)
    # ------------------------------------------------------------------
    print("\n[8] Saving profile plots to results/validation_single_step.png...")
    rho = state.rho

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Single-Step Validation: Transport Profiles after 1 step", fontsize=14)

    # Top-left: Te(rho)
    ax = axes[0, 0]
    ax.plot(rho, state.T_e, "r-", linewidth=2)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$T_e$ (keV)")
    ax.set_title("Electron Temperature")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Top-right: ne(rho)
    ax = axes[0, 1]
    ax.plot(rho, state.n_e, "b-", linewidth=2)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$n_e$ (m$^{-3}$)")
    ax.set_title("Electron Density")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Bottom-left: j_tor(rho)
    ax = axes[1, 0]
    ax.plot(rho, state.j_tor, "g-", linewidth=2)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$j_{tor}$ (A/m$^2$)")
    ax.set_title("Toroidal Current Density")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    # Bottom-right: q(rho)
    ax = axes[1, 1]
    ax.plot(rho, state.q, "m-", linewidth=2)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$q$")
    ax.set_title("Safety Factor")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/validation_single_step.png", dpi=150)
    plt.close(fig)
    print("  Plot saved.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a clear PASS/FAIL summary for all checks."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    n_pass = 0
    n_fail = 0
    for name, (passed, detail) in results.items():
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}: {detail}")

    print(f"\nTotal: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")
    if n_fail == 0:
        print("All checks passed.")
    else:
        print("Some checks FAILED -- review output above.")


if __name__ == "__main__":
    main()
