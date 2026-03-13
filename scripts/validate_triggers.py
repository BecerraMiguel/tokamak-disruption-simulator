"""
Validate disruption trigger detection.

Runs two disruptive scenarios (density and beta perturbations) through
the ShotRunner and verifies that the corresponding disruption triggers
fire as expected. The q95 trigger is skipped because baseline q95 is
already below 2.2 (known issue with minor radius).

Run from project root:
    python scripts/validate_triggers.py
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
    print("VALIDATION: Disruption trigger detection")
    print("=" * 60)

    from predisruption.iter_machine import build_iter_machine, ITER_PARAMS
    from predisruption.equilibrium import EquilibriumSolver
    from predisruption.transport import TransportSolver
    from predisruption.shot_runner import ShotRunner, disruptive_scenario, TRIGGERS

    results = {}

    # ------------------------------------------------------------------
    # 1. Density disruption
    # ------------------------------------------------------------------
    print("\n[1] Running density disruption scenario...")
    print(f"    perturbation_type='density', amp=0.4, t_end=75 s")

    tokamak1, _, domain1 = build_iter_machine(verbose=False)
    eq_solver1 = EquilibriumSolver(
        tokamak1, nx=65, ny=65,
        Rmin=domain1[0], Rmax=domain1[1],
        Zmin=domain1[2], Zmax=domain1[3],
    )
    tr_solver1 = TransportSolver(backend="simplified", n_rho=51)
    runner1 = ShotRunner(eq_solver1, tr_solver1, verbose=True)

    scenario_density = disruptive_scenario(
        perturbation_type="density",
        perturbation_amp=0.4,
        t_end=75.0,
        dt=1.0,
    )

    density_result = None
    try:
        density_result = runner1.run_shot(scenario_density, shot_id=1)
        d_label = density_result["label"]
        d_trigger = density_result["trigger"]
        d_time = density_result["disruption_time"]
        print(f"  label={d_label}, trigger={d_trigger}, disruption_time={d_time:.2f} s")

        density_trigger_ok = (d_label == 1 and d_trigger == "f_GW")
        results["Density shot triggers f_GW"] = (
            density_trigger_ok,
            f"label={d_label}, trigger={d_trigger}",
        )
    except Exception as exc:
        print(f"  ERROR: density shot failed: {exc}")
        import traceback
        traceback.print_exc()
        results["Density shot triggers f_GW"] = (False, f"Error: {exc}")

    # ------------------------------------------------------------------
    # 2. Beta disruption
    # ------------------------------------------------------------------
    print("\n[2] Running beta disruption scenario...")
    print(f"    perturbation_type='beta', amp=0.5, t_end=75 s")

    tokamak2, _, domain2 = build_iter_machine(verbose=False)
    eq_solver2 = EquilibriumSolver(
        tokamak2, nx=65, ny=65,
        Rmin=domain2[0], Rmax=domain2[1],
        Zmin=domain2[2], Zmax=domain2[3],
    )
    tr_solver2 = TransportSolver(backend="simplified", n_rho=51)
    runner2 = ShotRunner(eq_solver2, tr_solver2, verbose=True)

    scenario_beta = disruptive_scenario(
        perturbation_type="beta",
        perturbation_amp=0.5,
        t_end=75.0,
        dt=1.0,
    )

    beta_result = None
    try:
        beta_result = runner2.run_shot(scenario_beta, shot_id=2)
        b_label = beta_result["label"]
        b_trigger = beta_result["trigger"]
        b_time = beta_result["disruption_time"]
        print(f"  label={b_label}, trigger={b_trigger}, disruption_time={b_time:.2f} s")

        beta_trigger_ok = (b_label == 1 and b_trigger == "betaN")
        results["Beta shot triggers betaN"] = (
            beta_trigger_ok,
            f"label={b_label}, trigger={b_trigger}",
        )
    except Exception as exc:
        print(f"  ERROR: beta shot failed: {exc}")
        import traceback
        traceback.print_exc()
        results["Beta shot triggers betaN"] = (False, f"Error: {exc}")

    # ------------------------------------------------------------------
    # 3. Note about q95 trigger
    # ------------------------------------------------------------------
    print("\n[3] q95 trigger: SKIPPED")
    print("    Baseline q95 ~ 1.7 is already below the 2.2 threshold.")
    print("    This is a known issue: a_minor ~ 1.63 m < design 2.0 m,")
    print("    which lowers q95 proportionally. The q95 trigger cannot be")
    print("    tested independently until minor radius is resolved.")

    # ------------------------------------------------------------------
    # 4. Plot results (2x2 grid)
    # ------------------------------------------------------------------
    print("\n[4] Saving trigger validation plots to results/validation_triggers.png...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Disruption Trigger Validation", fontsize=14)

    # Top-left: density shot -- f_GW(t) with 0.95 limit
    ax = axes[0, 0]
    if density_result is not None:
        time_d = density_result["time"]
        ax.plot(time_d, density_result["f_GW"], "b-", linewidth=1.5, label="$f_{GW}$")
        ax.axhline(
            TRIGGERS["f_GW"], color="r", linestyle="--", linewidth=1,
            label=f"limit = {TRIGGERS['f_GW']}",
        )
        if density_result["label"] == 1:
            dt = density_result["disruption_time"]
            ax.axvline(dt, color="r", linewidth=2, alpha=0.7, label=f"disruption t={dt:.1f}s")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$f_{GW}$")
    ax.set_title("Density Shot: Greenwald Fraction")
    ax.grid(True, alpha=0.3)

    # Top-right: density shot -- betaN(t)
    ax = axes[0, 1]
    if density_result is not None:
        ax.plot(time_d, density_result["betaN"], "g-", linewidth=1.5, label=r"$\beta_N$")
        ax.axhline(
            TRIGGERS["betaN"], color="r", linestyle="--", linewidth=1,
            label=f"limit = {TRIGGERS['betaN']}",
        )
        if density_result["label"] == 1:
            dt = density_result["disruption_time"]
            ax.axvline(dt, color="r", linewidth=2, alpha=0.7)
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta_N$")
    ax.set_title("Density Shot: Normalised Beta")
    ax.grid(True, alpha=0.3)

    # Bottom-left: beta shot -- betaN(t) with 3.2 limit
    ax = axes[1, 0]
    if beta_result is not None:
        time_b = beta_result["time"]
        ax.plot(time_b, beta_result["betaN"], "g-", linewidth=1.5, label=r"$\beta_N$")
        ax.axhline(
            TRIGGERS["betaN"], color="r", linestyle="--", linewidth=1,
            label=f"limit = {TRIGGERS['betaN']}",
        )
        if beta_result["label"] == 1:
            bt = beta_result["disruption_time"]
            ax.axvline(bt, color="r", linewidth=2, alpha=0.7, label=f"disruption t={bt:.1f}s")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\beta_N$")
    ax.set_title("Beta Shot: Normalised Beta")
    ax.grid(True, alpha=0.3)

    # Bottom-right: beta shot -- f_GW(t)
    ax = axes[1, 1]
    if beta_result is not None:
        ax.plot(time_b, beta_result["f_GW"], "b-", linewidth=1.5, label="$f_{GW}$")
        ax.axhline(
            TRIGGERS["f_GW"], color="r", linestyle="--", linewidth=1,
            label=f"limit = {TRIGGERS['f_GW']}",
        )
        if beta_result["label"] == 1:
            bt = beta_result["disruption_time"]
            ax.axvline(bt, color="r", linewidth=2, alpha=0.7)
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$f_{GW}$")
    ax.set_title("Beta Shot: Greenwald Fraction")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/validation_triggers.png", dpi=150)
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
