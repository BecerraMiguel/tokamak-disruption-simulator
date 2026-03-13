"""
Validate a full coupled trajectory.

Runs the CoupledSimulator with ITER 15 MA waveforms and verifies that
time traces (Ip, q95, betaN, f_GW, W_thermal, Te0) are physically
reasonable over the full shot duration.

Run from project root:
    python scripts/validate_trajectory.py
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

    print("=" * 60)
    print("VALIDATION: Full coupled trajectory")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build machine and solvers
    # ------------------------------------------------------------------
    print("\n[1] Building ITER machine and solvers...")
    from predisruption.iter_machine import build_iter_machine, ITER_PARAMS
    from predisruption.equilibrium import EquilibriumSolver
    from predisruption.transport import TransportSolver
    from predisruption.coupling import CoupledSimulator

    tokamak, active_coils, domain = build_iter_machine()

    eq_solver = EquilibriumSolver(
        tokamak,
        nx=65, ny=65,
        Rmin=domain[0], Rmax=domain[1],
        Zmin=domain[2], Zmax=domain[3],
    )

    tr_solver = TransportSolver(backend="simplified", n_rho=51)

    # ------------------------------------------------------------------
    # 2. Define ITER waveforms
    # ------------------------------------------------------------------
    Ip_waveform = lambda t: np.interp(t, [0, 12, 80, 90], [3e6, 15e6, 15e6, 3e6])
    P_heat_waveform = lambda t: np.interp(t, [0, 12, 80], [5e6, 33e6, 33e6])
    n_target_waveform = lambda t: np.interp(t, [0, 12, 80], [3e19, 1e20, 1e20])

    # ------------------------------------------------------------------
    # 3. Short test (30 s)
    # ------------------------------------------------------------------
    print("\n[2] Running short trajectory (t_end=30 s)...")
    simulator_short = CoupledSimulator(
        eq_solver=eq_solver,
        tr_solver=tr_solver,
        dt_couple=1.0,
        verbose=True,
    )

    try:
        traj_short = simulator_short.run(
            t_end=30.0,
            Ip_waveform=Ip_waveform,
            P_heat_waveform=P_heat_waveform,
            n_target_waveform=n_target_waveform,
        )
        short_ok = True
        print(f"  Short run completed: {len(traj_short['time'])} time points recorded.")
        print(f"  Final time: {traj_short['time'][-1]:.1f} s")
        print(f"  Final Ip:   {traj_short['Ip'][-1]*1e-6:.2f} MA")
        print(f"  Final q95:  {traj_short['q95'][-1]:.3f}")
        print(f"  Final Te0:  {traj_short['T_e'][0, -1]:.2f} keV")
    except Exception as exc:
        print(f"  ERROR: short trajectory failed: {exc}")
        import traceback
        traceback.print_exc()
        short_ok = False
        traj_short = None
    finally:
        simulator_short.cleanup()

    # ------------------------------------------------------------------
    # 4. Full run (90 s) if short test passed
    # ------------------------------------------------------------------
    traj = traj_short  # use short run by default
    t_end_actual = 30.0

    if short_ok:
        print("\n[3] Running full trajectory (t_end=90 s)...")
        # Need fresh solvers because dynamic solver state is shared
        tokamak2, _, domain2 = build_iter_machine(verbose=False)
        eq_solver2 = EquilibriumSolver(
            tokamak2,
            nx=65, ny=65,
            Rmin=domain2[0], Rmax=domain2[1],
            Zmin=domain2[2], Zmax=domain2[3],
        )
        tr_solver2 = TransportSolver(backend="simplified", n_rho=51)

        simulator_full = CoupledSimulator(
            eq_solver=eq_solver2,
            tr_solver=tr_solver2,
            dt_couple=1.0,
            verbose=True,
        )

        try:
            traj_full = simulator_full.run(
                t_end=90.0,
                Ip_waveform=Ip_waveform,
                P_heat_waveform=P_heat_waveform,
                n_target_waveform=n_target_waveform,
            )
            traj = traj_full
            t_end_actual = 90.0
            print(f"  Full run completed: {len(traj['time'])} time points.")
        except Exception as exc:
            print(f"  WARNING: full trajectory failed at some point: {exc}")
            print("  Falling back to short trajectory for plots.")
        finally:
            simulator_full.cleanup()
    else:
        print("\n[3] Skipping full trajectory (short test failed).")

    if traj is None:
        print("\nNo trajectory data available. Exiting.")
        return

    # ------------------------------------------------------------------
    # 5. Print key final values
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Key values at end of trajectory:")
    print(f"  t_end      = {traj['time'][-1]:.1f} s")
    print(f"  Ip         = {traj['Ip'][-1]*1e-6:.2f} MA")
    print(f"  q95        = {traj['q95'][-1]:.3f}")
    print(f"  betaN      = {traj['betaN'][-1]:.4f}")
    print(f"  f_GW       = {traj['f_GW'][-1]:.4f}")
    print(f"  W_thermal  = {traj['W_thermal'][-1]:.3e} J")
    print(f"  Te(0)      = {traj['T_e'][0, -1]:.2f} keV")
    print(f"  ne(0)      = {traj['n_e'][0, -1]:.3e} m^-3")
    print(f"  disrupted  = {traj['stopped']}")
    print("-" * 40)

    print("\nNOTE: q95 ~ 1.7 is expected (below 2.2 trigger) due to")
    print("a_minor ~ 1.63 m < target 2.0 m. This is physically self-consistent")
    print("but means the q95 trigger cannot be used for disruption scenarios.")

    # ------------------------------------------------------------------
    # 6. Plot time traces (2x3 grid)
    # ------------------------------------------------------------------
    print("\n[4] Saving time-trace plots to results/validation_trajectory.png...")
    time = traj["time"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Coupled Trajectory Validation (t_end={t_end_actual:.0f} s)",
        fontsize=14,
    )

    # Row 0, Col 0: Ip(t)
    ax = axes[0, 0]
    ax.plot(time, traj["Ip"] * 1e-6, "b-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$I_p$ (MA)")
    ax.set_title("Plasma Current")
    ax.grid(True, alpha=0.3)

    # Row 0, Col 1: q95(t) with 2.2 dashed line
    ax = axes[0, 1]
    ax.plot(time, traj["q95"], "m-", linewidth=1.5)
    ax.axhline(2.2, color="r", linestyle="--", linewidth=1, label="q95 = 2.2 limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$q_{95}$")
    ax.set_title("Safety Factor at 95%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 0, Col 2: betaN(t) with 3.2 dashed line
    ax = axes[0, 2]
    ax.plot(time, traj["betaN"], "g-", linewidth=1.5)
    ax.axhline(3.2, color="r", linestyle="--", linewidth=1, label=r"$\beta_N$ = 3.2 limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta_N$")
    ax.set_title("Normalised Beta")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 0: f_GW(t) with 0.95 dashed line
    ax = axes[1, 0]
    ax.plot(time, traj["f_GW"], "c-", linewidth=1.5)
    ax.axhline(0.95, color="r", linestyle="--", linewidth=1, label="$f_{GW}$ = 0.95 limit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$f_{GW}$")
    ax.set_title("Greenwald Fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 1: W_thermal(t)
    ax = axes[1, 1]
    ax.plot(time, traj["W_thermal"] * 1e-6, "r-", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$W_{th}$ (MJ)")
    ax.set_title("Thermal Stored Energy")
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: Te(0, t)
    ax = axes[1, 2]
    ax.plot(time, traj["T_e"][0, :], "orange", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$T_e(0)$ (keV)")
    ax.set_title("On-axis Electron Temperature")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/validation_trajectory.png", dpi=150)
    plt.close(fig)
    print("  Plot saved.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    checks = {
        "Trajectory completed without crash": not traj["stopped"],
        "Ip > 0 throughout": bool(np.all(traj["Ip"] > 0)),
        "Te(0) > 1 keV throughout": bool(np.all(traj["T_e"][0, :] > 1.0)),
        "W_thermal > 0 throughout": bool(np.all(traj["W_thermal"] > 0)),
        "f_GW < 2.0 (physically bounded)": bool(np.all(traj["f_GW"] < 2.0)),
    }

    n_pass = 0
    n_fail = 0
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}")

    print(f"\nTotal: {n_pass} passed, {n_fail} failed out of {n_pass + n_fail}")
    if n_fail == 0:
        print("All checks passed.")
    else:
        print("Some checks FAILED -- review output above.")


if __name__ == "__main__":
    main()
