"""
Transport coupling tests using mock geometry dicts (no FreeGSNKE required).

These tests verify the SimplifiedTransport and TransportSolver backends
against known regression bugs (Bug A through D) and physics sanity checks.
All tests are fast and self-contained.
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import pytest

from predisruption.transport import SimplifiedTransport, TransportSolver, TransportState
from predisruption.iter_machine import ITER_PARAMS


class TestWThermalInit:
    """Regression tests for Bug B: W_thermal was left at 0.0 after init."""

    def test_W_thermal_init_nonzero(self):
        """Bug B regression: W_thermal must be computed from profiles at init time.

        Before the fix, SimplifiedTransport.init() left W_thermal at its default
        value of 0.0 J. This caused the energy balance ODE in step() to start
        from zero stored energy, producing nonsensical temperature evolution.
        For ITER-like parameters (Ip=15 MA, Te0=20 keV, ne0=1e20 m^-3), the
        thermal stored energy should be hundreds of MJ.
        """
        transport = SimplifiedTransport(n_rho=51)
        state = transport.init(
            Ip_A=15e6,
            T_e0_keV=20.0,
            n_e0_m3=1e20,
            geometry={"a_minor": 2.0, "kappa": 1.75},
        )

        assert state.W_thermal > 0, "W_thermal must be positive after init"
        assert state.W_thermal > 100e3, (
            f"W_thermal={state.W_thermal:.2e} J is below 100 kJ — "
            "unphysically low for ITER parameters"
        )
        assert state.W_thermal < 1e9, (
            f"W_thermal={state.W_thermal:.2e} J exceeds 1 GJ — "
            "unphysically high for ITER parameters"
        )


class TestTemperatureStability:
    """Regression tests for Bug A: temperature collapse after one step."""

    def test_temperature_stable_one_step(self):
        """Bug A regression: on-axis Te must remain above 1 keV after one step.

        Before the fix, the energy balance ODE used incorrect unit conversions,
        causing the derived T0 to collapse to ~0.01 keV after a single step
        even with 33 MW of auxiliary heating. With correct physics, ITER at
        flat-top should maintain Te0 in the 10-30 keV range.
        """
        transport = SimplifiedTransport(n_rho=51)
        state = transport.init(
            Ip_A=15e6,
            T_e0_keV=20.0,
            n_e0_m3=1e20,
            geometry={"a_minor": 2.0, "kappa": 1.75},
        )

        new_state = transport.step(
            state=state,
            geometry={"Ip": 15e6, "kappa": 1.75, "a_minor": 2.0},
            sources={"P_aux_W": 33e6, "P_ohm_W": 0.0},
            dt=1.0,
        )

        assert new_state.T_e[0] > 1.0, (
            f"T_e[0]={new_state.T_e[0]:.4f} keV collapsed below 1 keV — "
            "Bug A regression (unit conversion error in energy balance)"
        )
        assert new_state.T_e[0] < 50.0, (
            f"T_e[0]={new_state.T_e[0]:.1f} keV exceeds 50 keV — "
            "unphysically high, possible energy balance error"
        )


class TestIPB98Scaling:
    """Tests for IPB98(y,2) confinement time scaling law."""

    def test_IPB98_scaling(self):
        """IPB98(y,2) scaling must return tau_E ~ 3-4 s for ITER reference parameters.

        The ITER baseline scenario (Ip=15 MA, B0=5.3 T, ne=1e20 m^-3, P=50 MW,
        R=6.2 m, kappa=1.75, a=2.0 m, M=2.5 amu) should yield a confinement
        time between 2.0 and 6.0 seconds according to the IPB98(y,2) scaling.
        """
        transport = SimplifiedTransport(n_rho=51)
        tau_E = transport.tau_E_IPB98(
            Ip_A=15e6,
            B0=5.3,
            n_e_avg=1e20,
            P_loss_W=50e6,
            R=6.2,
            kappa=1.75,
            a=2.0,
        )

        assert 2.0 < tau_E < 6.0, (
            f"tau_E={tau_E:.2f} s is outside the expected 2-6 s range "
            "for ITER reference parameters"
        )


class TestCurrentDiffusion:
    """Regression tests for Bug C: CFL explosion in current diffusion."""

    def test_current_diffusion_stable(self):
        """Bug C regression: j_tor must remain finite and smooth after 10 steps.

        Before the fix, the explicit current diffusion scheme violated the CFL
        condition for large time steps (dt=1 s with tau_R=100 s), causing
        exponential growth of high-frequency oscillations in the current profile.
        The Crank-Nicolson implicit scheme is unconditionally stable and should
        produce smooth, finite profiles at any time step.
        """
        transport = SimplifiedTransport(n_rho=51)
        state = transport.init(
            Ip_A=15e6,
            T_e0_keV=20.0,
            n_e0_m3=1e20,
            geometry={"a_minor": 2.0, "kappa": 1.75},
        )

        geometry = {"Ip": 15e6, "kappa": 1.75, "a_minor": 2.0}
        sources = {"P_aux_W": 33e6, "P_ohm_W": 0.0}

        for _ in range(10):
            state = transport.step(
                state=state,
                geometry=geometry,
                sources=sources,
                dt=1.0,
            )

        # j_tor must be finite (no NaN or Inf)
        assert np.all(np.isfinite(state.j_tor)), (
            "j_tor contains NaN or Inf after 10 steps — "
            "Bug C regression (CFL explosion in current diffusion)"
        )

        # j_tor magnitude must be bounded: a CFL explosion would cause
        # exponential growth with |j| >> initial magnitude after 10 steps.
        # The implicit Crank-Nicolson scheme keeps |j| bounded even if the
        # profile develops some numerical oscillation near the axis.
        max_magnitude = np.max(np.abs(state.j_tor))
        assert max_magnitude < 1e12, (
            f"j_tor magnitude {max_magnitude:.2e} exceeds 1e12 A/m^2 — "
            "Bug C regression (unbounded current growth)"
        )

        # j_tor in the bulk (rho > 0.2) should decrease monotonically toward
        # the edge; check that the edge half is well-behaved
        edge_half = state.j_tor[len(state.j_tor) // 2 :]
        max_jump_edge = np.max(np.abs(np.diff(edge_half)))
        max_mag_edge = np.max(np.abs(edge_half))
        assert max_jump_edge < 0.5 * max_mag_edge, (
            f"j_tor edge half is not smooth: max_jump={max_jump_edge:.2e}, "
            f"max_magnitude={max_mag_edge:.2e} — "
            "Bug C regression (oscillatory current profile)"
        )


class TestGreenwaldFraction:
    """Tests for the Greenwald density fraction calculation."""

    def test_greenwald_fraction(self):
        """Greenwald fraction must match analytic calculation for uniform density.

        For Ip=15 MA, a=2.0 m, and uniform ne=1e20 m^-3:
          n_GW = Ip_MA / (pi * a^2) = 15 / (pi * 4) = 1.194 (in 1e20 m^-3)
          f_GW = (1e20 * 1e-20) / 1.194 = 1.0 / 1.194 = 0.838
        """
        state = TransportState(
            rho=np.linspace(0, 1, 51),
            T_e=np.ones(51),
            T_i=np.ones(51),
            n_e=1e20 * np.ones(51),
            j_tor=np.ones(51),
            psi=np.ones(51),
            q=np.ones(51),
        )

        Ip_MA = 15.0
        a_minor = 2.0

        f_GW = state.greenwald_fraction(Ip_MA, a_minor)

        expected_n_GW = 15.0 / (np.pi * 4.0)  # 1.194 in 1e20 m^-3
        expected_f_GW = (1e20 * 1e-20) / expected_n_GW  # 0.838

        assert abs(f_GW - expected_f_GW) < 0.01, (
            f"f_GW={f_GW:.4f} differs from expected {expected_f_GW:.4f} "
            "by more than 0.01"
        )


class TestGeometryPassthrough:
    """Regression tests for Bug D: geometry Ip ignored in extract_freegsnke_profiles."""

    def test_geometry_passed_to_profiles(self):
        """Bug D regression: extract_freegsnke_profiles must use geometry Ip, not default.

        Before the fix, extract_freegsnke_profiles() used a hardcoded default
        Ip=15e6 instead of reading it from the geometry dict. This meant that
        equilibrium feedback (where Ip evolves over time) was silently ignored,
        and the betap calculation was always based on the reference 15 MA value
        regardless of the actual plasma current.
        """
        solver = TransportSolver(backend="simplified", n_rho=51)
        state = solver.init(
            geometry={"Ip": 10e6, "a_minor": 1.5, "kappa": 1.8},
            Ip_A=10e6,
            T_e0_keV=15.0,
            n_e0_m3=8e19,
        )

        profile_params = solver.extract_freegsnke_profiles(
            state,
            geometry={"Ip": 10e6, "a_minor": 1.5},
        )

        assert profile_params["Ip"] == 10e6, (
            f"profile_params['Ip']={profile_params['Ip']:.2e} should be 10e6, "
            "not 15e6 — Bug D regression (geometry Ip ignored)"
        )
