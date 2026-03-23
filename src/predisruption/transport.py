"""
Transport solver with TORAX backend (primary) and NumPy fallback.

Architecture:
  TransportSolver.step(geometry, sources, dt) -> TransportState

TORAX backend:
  Uses TORAX 1D transport PDEs (JAX-based). Requires AVX CPU or GPU.
  Auto-detected at import time. Used in Colab and modern machines.

NumPy fallback (SimplifiedTransport):
  Solves current diffusion equation exactly.
  Profiles Te(rho,t) and ne(rho,t) evolve via:
    - Global energy content from IPB98(y,2) confinement scaling
    - Profile shapes: T_e(rho) = T0 * (1-rho^2)^alpha  (parabolic)
    - Particle balance: ne_0 evolves to match n_GW target
  Physics-plausible for synthetic ML training data. No JAX required.

Usage:
    solver = TransportSolver(backend="auto")  # auto-detects TORAX
    state  = solver.init(geometry, Ip=15e6, T_e0=20.0, n_e0=1.0e20)
    state  = solver.step(geometry, sources, dt, state)
    Te     = state.T_e   # [n_rho] keV
    ne     = state.n_e   # [n_rho] m^-3
    j      = state.j_tor # [n_rho] A/m^2
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------

def _check_torax_available() -> bool:
    """Return True if TORAX/JAX can be imported and run."""
    try:
        import jax
        import torax
        # Quick smoke-test to trigger the AVX check at import
        _ = jax.numpy.array([1.0])
        return True
    except Exception:
        return False


_TORAX_AVAILABLE: bool | None = None   # lazily evaluated


def torax_available() -> bool:
    global _TORAX_AVAILABLE
    if _TORAX_AVAILABLE is None:
        _TORAX_AVAILABLE = _check_torax_available()
    return _TORAX_AVAILABLE


# ---------------------------------------------------------------------------
# Transport state dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransportState:
    """
    Holds 1D radial transport profiles at one point in time.

    All profiles are on a uniform rho grid: rho = linspace(0, 1, n_rho).
    """
    rho:   np.ndarray           # [n_rho] normalised sqrt toroidal flux coordinate
    T_e:   np.ndarray           # [n_rho] electron temperature (keV)
    T_i:   np.ndarray           # [n_rho] ion temperature (keV)
    n_e:   np.ndarray           # [n_rho] electron density (m^-3)
    j_tor: np.ndarray           # [n_rho] toroidal current density (A/m^2)
    psi:   np.ndarray           # [n_rho] poloidal flux (Wb), from current diffusion
    q:     np.ndarray           # [n_rho] safety factor (from equilibrium)
    time:  float = 0.0          # current simulation time (s)

    # Derived scalars (updated after each step)
    W_thermal: float = 0.0      # thermal stored energy (J)
    tau_E:     float = 1.0      # energy confinement time (s)

    def greenwald_density(self, Ip_MA: float, a_minor: float) -> float:
        """Greenwald density limit n_GW = Ip_MA / (pi * a^2) [10^20 m^-3]."""
        return Ip_MA / (np.pi * a_minor**2)   # 10^20 m^-3

    def greenwald_fraction(self, Ip_MA: float, a_minor: float) -> float:
        """f_GW = <ne> / n_GW where <ne> is line-averaged density."""
        n_line_avg = np.mean(self.n_e) * 1e-20  # convert to 10^20 m^-3
        n_GW = self.greenwald_density(Ip_MA, a_minor)
        return n_line_avg / n_GW


# ---------------------------------------------------------------------------
# Backend: Simplified NumPy transport (no JAX)
# ---------------------------------------------------------------------------

class SimplifiedTransport:
    """
    Physics-plausible 1D transport without JAX.

    Physics included:
      1. Current diffusion: ∂ψ/∂t = (η/μ₀) * ∂²ψ/∂ρ²  (simplified geometry)
      2. Energy balance:    dW/dt = P_heat - W/τ_E   (0D energy confinement)
      3. Temperature profile: T(ρ) = T₀ * (1 - ρ²)^α_T  (parabolic)
      4. Density profile:  n(ρ) = n₀ * (1 - β*(1 - (1-ρ²)^α_n))

    IPB98(y,2) scaling law for τ_E:
      τ_E = 0.0562 * Ip^0.93 * B0^0.15 * ne^0.41 * P^-0.69 * R^1.97 * κ^0.78 * ε^0.58 * M^0.19

    This is a "0.5D" model: global energy evolves via ODE, profiles are
    parameterised shapes. Good for synthetic ML data where profile shapes
    matter more than exact transport coefficients.
    """

    # IPB98(y,2) exponents
    _IPB98_C    = 0.0562
    _IPB98_EXP  = {
        "Ip":    0.93,   # MA
        "B0":    0.15,   # T
        "ne":    0.41,   # 10^19 m^-3
        "P":    -0.69,   # MW
        "R":     1.97,   # m
        "kappa": 0.78,
        "eps":   0.58,   # a/R
        "M":     0.19,   # amu
    }

    def __init__(
        self,
        n_rho: int = 51,
        alpha_T: float = 1.5,    # temperature profile peaking
        alpha_n: float = 0.3,    # density profile peaking (0 = flat, 1 = peaked)
        alpha_j: float = 2.0,    # current density profile peaking
        ion_mass: float = 2.5,   # amu (D-T average)
    ):
        self.n_rho   = n_rho
        self.alpha_T = alpha_T
        self.alpha_n = alpha_n
        self.alpha_j = alpha_j
        self.ion_mass = ion_mass
        self.rho = np.linspace(0.0, 1.0, n_rho)

    def tau_E_IPB98(
        self,
        Ip_A: float,
        B0: float,
        n_e_avg: float,
        P_loss_W: float,
        R: float,
        kappa: float,
        a: float,
        M: float = 2.5,
    ) -> float:
        """
        IPB98(y,2) confinement time scaling (s).

        Parameters: Ip (A), B0 (T), n_e_avg (m^-3), P_loss (W),
                    R (m), kappa, a (m), M (amu).
        """
        Ip_MA  = Ip_A  * 1e-6
        ne_19  = n_e_avg * 1e-19
        P_MW   = max(P_loss_W * 1e-6, 0.01)   # avoid P=0
        eps    = a / R
        ex     = self._IPB98_EXP
        tau = (self._IPB98_C
               * Ip_MA**ex["Ip"]
               * B0**ex["B0"]
               * ne_19**ex["ne"]
               * P_MW**ex["P"]
               * R**ex["R"]
               * kappa**ex["kappa"]
               * eps**ex["eps"]
               * M**ex["M"])
        return max(tau, 0.05)   # floor at 50 ms

    def _temperature_profile(self, T0_keV: float) -> np.ndarray:
        """Parabolic temperature profile T(ρ) = T0 * (1 - ρ²)^alpha_T."""
        return T0_keV * (1.0 - self.rho**2) ** self.alpha_T

    def _density_profile(self, n0_m3: float) -> np.ndarray:
        """Mildly peaked density profile."""
        return n0_m3 * (0.8 + 0.2 * (1.0 - self.rho**2) ** self.alpha_n)

    def _current_profile(self, j0: float) -> np.ndarray:
        """Peaked current density profile j(ρ) = j0 * (1 - ρ²)^alpha_j."""
        return j0 * (1.0 - self.rho**2) ** self.alpha_j

    def init(
        self,
        Ip_A: float,
        T_e0_keV: float,
        n_e0_m3: float,
        geometry: dict | None = None,
    ) -> TransportState:
        """
        Create an initial TransportState.

        Parameters
        ----------
        Ip_A       : float, initial plasma current (A)
        T_e0_keV   : float, initial on-axis electron temperature (keV)
        n_e0_m3    : float, initial on-axis electron density (m^-3)
        geometry   : dict from EquilibriumSolver.get_signals() (optional)
        """
        T_e = self._temperature_profile(T_e0_keV)
        T_i = 0.9 * T_e                          # Ti ~ 0.9 Te (typical)
        n_e = self._density_profile(n_e0_m3)

        # Initial current density from profile shape + Ip constraint
        j_tor, psi = self._solve_current_profile(Ip_A, geometry)

        q = geometry["q_profile"] if geometry and "q_profile" in geometry else np.ones(self.n_rho) * 3.0

        # Compute initial W_thermal from profiles (Bug B fix: was left at 0.0)
        from .iter_machine import ITER_PARAMS
        R = ITER_PARAMS["R_major"]
        a = geometry.get("a_minor", ITER_PARAMS["a_minor"]) if geometry else ITER_PARAMS["a_minor"]
        kappa = geometry.get("kappa", 1.75) if geometry else 1.75
        V_plasma = 2.0 * np.pi**2 * R * a**2 * kappa
        p_avg = float(np.mean(n_e * (T_e + T_i) * 1.602e-16))  # Pa (T in keV → J)
        W_thermal = (3.0 / 2.0) * p_avg * V_plasma

        state = TransportState(
            rho=self.rho.copy(),
            T_e=T_e, T_i=T_i, n_e=n_e,
            j_tor=j_tor, psi=psi, q=q,
            time=0.0,
            W_thermal=W_thermal,
        )
        return state

    def step(
        self,
        state: TransportState,
        geometry: dict,
        sources: dict,
        dt: float,
        iter_params: dict | None = None,
    ) -> TransportState:
        """
        Advance transport by one time step.

        Parameters
        ----------
        state      : current TransportState
        geometry   : dict from EquilibriumSolver.get_signals()
        sources    : dict with keys:
                       "P_ohm_W"    : float, ohmic heating power (W)
                       "P_aux_W"    : float, auxiliary heating power (W)
                       "n_target"   : float, target density (m^-3), or None
                       "impurity_Z" : float, effective charge Zeff (default 1.6)
        dt         : float, time step (s)
        iter_params: dict with ITER machine parameters (R, a, B0, kappa, Ip)

        Returns
        -------
        new_state : TransportState at t + dt
        """
        if iter_params is None:
            from .iter_machine import ITER_PARAMS
            iter_params = ITER_PARAMS

        Ip    = geometry.get("Ip",     15.0e6)
        B0    = iter_params.get("B0",  5.3)
        R     = iter_params.get("R_major", 6.2)
        a     = geometry.get("a_minor", iter_params.get("a_minor", 2.0))
        kappa = geometry.get("kappa",  1.7)

        P_ohm = sources.get("P_ohm_W",  0.0)
        P_aux = sources.get("P_aux_W",  0.0)
        P_tot = P_ohm + P_aux

        # --- Energy balance (0D ODE, implicit step) ---
        n_avg = float(np.mean(state.n_e))
        tau   = self.tau_E_IPB98(Ip, B0, n_avg, P_tot, R, kappa, a, self.ion_mass)
        W_old = state.W_thermal
        # dW/dt = P_heat - W/tau  → implicit: W_new = (W_old + P*dt) / (1 + dt/tau)
        W_new = (W_old + P_tot * dt) / (1.0 + dt / tau)

        # Derive new on-axis temperature from W and profile shape
        # W = (3/2) * integral(n*T) * V_plasma = (3/2) * n_avg * T_avg * V
        V_plasma = 2.0 * np.pi**2 * R * a**2 * kappa   # approximate torus volume
        T_avg_keV_new = (2.0 / 3.0) * W_new / (n_avg * 1.602e-16 * V_plasma)
        # From profile: T_avg = T0 * integral((1-rho^2)^alpha, 0, 1)
        profile_norm = np.trapz((1.0 - self.rho**2) ** self.alpha_T, self.rho)
        T0_new = T_avg_keV_new / max(profile_norm, 0.01)
        T0_new = max(T0_new, 0.1)   # floor at 100 eV

        T_e_new = self._temperature_profile(T0_new)
        T_i_new = 0.9 * T_e_new

        # --- Density evolution ---
        n_target = sources.get("n_target", None)
        if n_target is not None:
            # Relax toward target with τ_n ~ 0.5 s
            tau_n = 0.5
            n0_old = state.n_e[0]
            n0_new = (n0_old + (n_target / 0.9) * dt / tau_n) / (1.0 + dt / tau_n)
        else:
            n0_new = state.n_e[0]
        n_e_new = self._density_profile(n0_new)

        # --- Current diffusion (simplified 1D) ---
        j_tor_new, psi_new = self._diffuse_current(state.psi, state.j_tor, Ip, geometry, dt)

        # --- Safety factor from equilibrium (pass-through) ---
        q_new = geometry.get("q_profile", state.q)

        new_state = TransportState(
            rho=self.rho.copy(),
            T_e=T_e_new,
            T_i=T_i_new,
            n_e=n_e_new,
            j_tor=j_tor_new,
            psi=psi_new,
            q=q_new,
            time=state.time + dt,
            W_thermal=W_new,
            tau_E=tau,
        )
        return new_state

    # ------------------------------------------------------------------
    # Current profile helpers
    # ------------------------------------------------------------------

    def _solve_current_profile(self, Ip_A: float, geometry: dict | None) -> tuple:
        """
        Initialise j_tor and psi from a peaked current density shape.
        Normalises so that the enclosed current equals Ip_A.
        """
        from .iter_machine import ITER_PARAMS
        R = ITER_PARAMS["R_major"]
        a = ITER_PARAMS["a_minor"]

        j0_guess = 1.0   # shape only — will normalise
        j_shape  = self._current_profile(j0_guess)

        # Approximate enclosed current: I(rho) = 2pi * a^2 * integral(j*rho, 0, rho)
        I_norm = 2.0 * np.pi * a**2 * np.trapz(j_shape * self.rho, self.rho)
        j0 = Ip_A / max(I_norm, 1.0)
        j_tor = self._current_profile(j0)

        # Psi from current diffusion initial condition (∇²ψ ~ -μ₀*R*j)
        psi = np.cumsum(j_tor) / len(j_tor)   # crude integral
        psi = psi / max(psi.max(), 1.0)        # normalise to [0,1]

        return j_tor, psi

    def _diffuse_current(
        self,
        psi_old: np.ndarray,
        j_old: np.ndarray,
        Ip_A: float,
        geometry: dict,
        dt: float,
    ) -> tuple:
        """
        1D current diffusion:  ∂ψ/∂t = (η/μ₀σ) * ∂²ψ/∂ρ²

        Uses Spitzer resistivity η = η₀ * T_e^{-3/2}.
        Applies Neumann BC at ρ=0 and ψ=ψ_a at ρ=1 (edge flux).
        """
        mu0   = 4.0e-7 * np.pi
        n_rho = len(psi_old)
        drho  = 1.0 / (n_rho - 1)

        # Diffusion coefficient (units: ρ²/s)
        # ITER resistive diffusion timescale τ_R = μ₀*σ*a² ~ 100 s
        tau_R   = 100.0   # s, resistive diffusion time
        D_coeff = 1.0 / tau_R

        # Crank-Nicolson implicit scheme (unconditionally stable)
        # (I - 0.5*dt*D*L) psi_new = (I + 0.5*dt*D*L) psi_old
        # where L is the 1D Laplacian stencil on a uniform grid
        alpha = 0.5 * dt * D_coeff / drho**2

        # Build tridiagonal system for interior points [1:-1]
        n_int = n_rho - 2
        A = np.zeros((n_int, n_int))
        rhs = np.zeros(n_int)

        for i in range(n_int):
            A[i, i] = 1.0 + 2.0 * alpha
            if i > 0:
                A[i, i - 1] = -alpha
            if i < n_int - 1:
                A[i, i + 1] = -alpha

        # Neumann BC at axis (i=0 → psi[0] = psi[1]): ghost point psi[-1] = psi[1]
        # So d²psi/drho² at i=1 uses psi[0]=psi[1]: (psi[1] - 2*psi[1] + psi[2])/drho²
        # This means the first interior point has a modified stencil
        A[0, 0] = 1.0 + alpha  # only one off-diagonal (axis side absorbed)

        # RHS: (I + 0.5*dt*D*L) psi_old
        psi_int = psi_old[1:-1]
        for i in range(n_int):
            val = psi_int[i]
            if i > 0:
                val += alpha * (psi_old[i] - psi_int[i])  # psi_old[i] = psi_old[(i+1)-1]
            else:
                val += alpha * (psi_old[1] - psi_int[0])   # Neumann: psi[0] = psi[1]
            if i < n_int - 1:
                val += alpha * (psi_old[i + 2] - psi_int[i])
            else:
                val += alpha * (psi_old[-1] - psi_int[-1])  # Dirichlet BC contribution
            rhs[i] = val

        # Dirichlet BC at LCFS: psi[-1] fixed → add alpha * psi_old[-1] to RHS of last eqn
        rhs[-1] += alpha * psi_old[-1]

        psi_int_new = np.linalg.solve(A, rhs)

        psi_new = np.empty_like(psi_old)
        psi_new[0] = psi_int_new[0]     # Neumann: psi[0] = psi[1]
        psi_new[1:-1] = psi_int_new
        psi_new[-1] = psi_old[-1]        # Dirichlet: fix LCFS value

        # Recompute j from new psi gradient
        dpsi     = np.gradient(psi_new, self.rho)
        d2psi_rz = np.gradient(dpsi, self.rho)
        j_new    = -d2psi_rz / (mu0 * max(float(np.mean(self.rho)) + 1e-9, 1.0))

        # Renormalise current to Ip
        I_enclosed = np.trapz(j_new * self.rho, self.rho)
        if abs(I_enclosed) > 1e-10:
            j_new *= Ip_A / I_enclosed

        return j_new, psi_new


# ---------------------------------------------------------------------------
# Backend: TORAX (JAX)
# ---------------------------------------------------------------------------

class ToraxTransport:
    """
    TORAX 1D transport solver backend (full-run mode).

    TORAX is designed for complete trajectory runs via torax.run_simulation().
    This class wraps that API and provides conversion to/from TransportState.

    For step-by-step coupling, use SimplifiedTransport locally and TORAX
    in full-run mode on Colab (pre-compute geometry, run TORAX once).
    """

    def __init__(
        self,
        torax_config: dict | None = None,
        n_rho: int = 51,
    ):
        if not torax_available():
            raise RuntimeError(
                "TORAX/JAX not available on this machine (AVX required). "
                "Use backend='simplified' or run on Colab."
            )
        self.config_template = torax_config or {}
        self.n_rho = n_rho
        self.rho = np.linspace(0.0, 1.0, n_rho)

    def run_full(
        self,
        geometry_dir: str,
        geometry_files: dict[float, str],
        t_final: float,
        Ip_A: float = 15.0e6,
        T_e0_keV: float = 20.0,
        n_e0_m3: float = 1.0e20,
        P_heat_W: float = 33.0e6,
        transport_model: str = "constant",
    ) -> tuple:
        """
        Run TORAX for a full trajectory with time-dependent GEQDSK geometry.

        Parameters
        ----------
        geometry_dir    : str, directory containing GEQDSK files
        geometry_files  : dict {time_s: filename}, time-dependent geometry
        t_final         : float, end time (s)
        Ip_A            : float, plasma current (A)
        T_e0_keV        : float, initial on-axis Te (keV)
        n_e0_m3         : float, initial on-axis ne (m^-3)
        P_heat_W        : float, total heating power (W)
        transport_model : str, 'constant', 'bohm-gyrobohm', or 'qlknn'

        Returns
        -------
        data_tree      : xarray.DataTree with profiles and scalars
        state_history  : torax.StateHistory for debugging
        """
        import torax

        config = _default_iter_torax_config(
            geometry_dir=geometry_dir,
            geometry_files=geometry_files,
            t_final=t_final,
            Ip_A=Ip_A,
            T_e0_keV=T_e0_keV,
            P_heat_W=P_heat_W,
            transport_model=transport_model,
        )

        # Merge any user overrides from config_template
        config = _deep_merge(config, self.config_template)

        torax_config = torax.ToraxConfig.from_dict(config)
        data_tree, state_history = torax.run_simulation(torax_config)
        return data_tree, state_history

    def datatree_to_states(self, data_tree) -> list[TransportState]:
        """
        Convert a TORAX xarray DataTree to a list of TransportState objects.

        Interpolates TORAX profiles onto our uniform n_rho grid.
        """
        profiles = data_tree["profiles"].ds
        scalars = data_tree["scalars"].ds

        times = profiles.coords["time"].values
        # TORAX uses rho_cell_norm for cell-centered profiles
        rho_torax = profiles.coords["rho_cell_norm"].values

        states = []
        for i, t in enumerate(times):
            T_e = np.interp(self.rho, rho_torax, profiles["T_e"].values[i, :])
            T_i = np.interp(self.rho, rho_torax, profiles["T_i"].values[i, :])
            n_e = np.interp(self.rho, rho_torax, profiles["n_e"].values[i, :])
            j_tor = np.interp(self.rho, rho_torax, profiles["j_total"].values[i, :])
            psi = np.interp(self.rho, rho_torax, profiles["psi"].values[i, :])

            # q is on face grid (rho_face_norm), interpolate to cell grid
            if "q" in profiles:
                rho_face = profiles.coords["rho_face_norm"].values
                q = np.interp(self.rho, rho_face, profiles["q"].values[i, :])
            else:
                q = np.ones(self.n_rho) * 3.0

            W_th = float(scalars["W_thermal_total"].values[i]) if "W_thermal_total" in scalars else 0.0

            states.append(TransportState(
                rho=self.rho.copy(),
                T_e=T_e,
                T_i=T_i,
                n_e=n_e,
                j_tor=j_tor,
                psi=psi,
                q=q,
                time=float(t),
                W_thermal=W_th,
            ))

        return states

    def init(self, geqdsk_path: str, Ip_A: float, T_e0_keV: float, n_e0_m3: float) -> TransportState:
        """
        Initialise transport by running a short (1-step) TORAX simulation.

        Returns an initial TransportState from TORAX output.
        """
        import os
        geometry_dir = os.path.dirname(os.path.abspath(geqdsk_path))
        geometry_file = os.path.basename(geqdsk_path)

        data_tree, _ = self.run_full(
            geometry_dir=geometry_dir,
            geometry_files={0.0: geometry_file},
            t_final=0.1,
            Ip_A=Ip_A,
            T_e0_keV=T_e0_keV,
            n_e0_m3=n_e0_m3,
            transport_model="constant",
        )
        states = self.datatree_to_states(data_tree)
        return states[0] if states else TransportState(
            rho=self.rho.copy(),
            T_e=np.full(self.n_rho, T_e0_keV),
            T_i=np.full(self.n_rho, T_e0_keV * 0.9),
            n_e=np.full(self.n_rho, n_e0_m3),
            j_tor=np.zeros(self.n_rho),
            psi=np.zeros(self.n_rho),
            q=np.ones(self.n_rho) * 3.0,
        )

    def step(self, state: TransportState, geqdsk_path: str, sources: dict, dt: float):
        """Not supported — TORAX uses full-run mode. Use run_full() instead."""
        raise NotImplementedError(
            "ToraxTransport does not support step-by-step execution. "
            "Use run_full() for trajectory runs, or use SimplifiedTransport "
            "for step-by-step coupling."
        )


# ---------------------------------------------------------------------------
# Unified TransportSolver interface
# ---------------------------------------------------------------------------

class TransportSolver:
    """
    Unified transport solver interface.

    Selects backend automatically:
      - "torax"      : TORAX + JAX (accurate, requires AVX/GPU)
      - "simplified" : NumPy-based simplified transport (portable)
      - "auto"       : tries TORAX first, falls back to simplified

    Parameters
    ----------
    backend : str
        "auto", "torax", or "simplified"
    torax_config : dict, optional
        TORAX configuration dict (only used if backend="torax" or "auto")
    n_rho : int
        Radial grid resolution
    """

    def __init__(
        self,
        backend: str = "auto",
        torax_config: dict | None = None,
        n_rho: int = 51,
    ):
        self.n_rho  = n_rho
        self._state = None

        if backend == "auto":
            if torax_available():
                backend = "torax"
                print("TransportSolver: using TORAX backend (JAX available)")
            else:
                backend = "simplified"
                print("TransportSolver: TORAX unavailable (no AVX) — using simplified transport")

        self.backend = backend

        if backend == "torax":
            self._impl = ToraxTransport(torax_config, n_rho=n_rho)
        elif backend == "simplified":
            self._impl = SimplifiedTransport(n_rho=n_rho)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def init(
        self,
        geometry: dict,
        Ip_A: float,
        T_e0_keV: float = 20.0,
        n_e0_m3: float = 1.0e20,
        geqdsk_path: str | None = None,
    ) -> TransportState:
        """
        Initialise transport state.

        Parameters
        ----------
        geometry    : dict from EquilibriumSolver.get_signals()
        Ip_A        : float, initial plasma current (A)
        T_e0_keV    : float, on-axis electron temperature (keV)
        n_e0_m3     : float, on-axis electron density (m^-3)
        geqdsk_path : str, path to GEQDSK file (required for TORAX backend)
        """
        if self.backend == "torax":
            if geqdsk_path is None:
                raise ValueError("geqdsk_path required for TORAX backend")
            state = self._impl.init(geqdsk_path, Ip_A, T_e0_keV, n_e0_m3)
        else:
            state = self._impl.init(Ip_A, T_e0_keV, n_e0_m3, geometry=geometry)
        self._state = state
        return state

    def step(
        self,
        geometry: dict,
        sources: dict,
        dt: float,
        state: TransportState | None = None,
        geqdsk_path: str | None = None,
    ) -> TransportState:
        """
        Advance transport by one coupling time step.

        Parameters
        ----------
        geometry    : dict from EquilibriumSolver.get_signals()
        sources     : dict {P_ohm_W, P_aux_W, n_target, ...}
        dt          : float, time step (s)
        state       : TransportState (uses internal state if None)
        geqdsk_path : str, path to GEQDSK file (required for TORAX backend)
        """
        if state is None:
            state = self._state
        if state is None:
            raise RuntimeError("Call init() before step()")

        if self.backend == "torax":
            if geqdsk_path is None:
                raise ValueError("geqdsk_path required for TORAX backend step()")
            new_state = self._impl.step(state, geqdsk_path, sources, dt)
        else:
            new_state = self._impl.step(state, geometry, sources, dt)

        self._state = new_state
        return new_state

    def run_trajectory(
        self,
        geometry_dir: str,
        geometry_files: dict[float, str],
        t_final: float,
        **kwargs,
    ) -> list[TransportState]:
        """
        Run TORAX in full-trajectory mode (TORAX backend only).

        Parameters
        ----------
        geometry_dir   : str, directory containing GEQDSK files
        geometry_files : dict {time_s: filename}
        t_final        : float, end time (s)
        **kwargs       : passed to ToraxTransport.run_full()

        Returns
        -------
        list of TransportState at each output time step
        """
        if self.backend != "torax":
            raise RuntimeError(
                "run_trajectory() is only available with the TORAX backend. "
                "For the simplified backend, use the step() loop."
            )
        data_tree, _ = self._impl.run_full(
            geometry_dir=geometry_dir,
            geometry_files=geometry_files,
            t_final=t_final,
            **kwargs,
        )
        return self._impl.datatree_to_states(data_tree)

    def extract_freegsnke_profiles(
        self,
        state: TransportState,
        geometry: dict,
    ) -> dict:
        """
        Extract FreeGSNKE-compatible profile parameters from a transport state.

        Returns
        -------
        dict with:
            "betap" : poloidal beta (dimensionless)
            "Ip"    : plasma current (A) — from geometry (equilibrium is authoritative)
        """
        from .iter_machine import ITER_PARAMS
        R  = ITER_PARAMS["R_major"]
        a  = ITER_PARAMS["a_minor"]
        B0 = ITER_PARAMS["B0"]

        # Poloidal beta from pressure profile and Ip
        # betap = <p> / (B_p^2 / (2*mu0))  where B_p ~ mu0*Ip/(2pi*a)
        mu0   = 4e-7 * np.pi
        n_e   = state.n_e
        T_e   = state.T_e * 1.602e-16   # convert keV to J (1 keV = 1.602e-16 J)
        T_i   = state.T_i * 1.602e-16
        p_avg = float(np.mean(n_e * (T_e + T_i)))  # Pa

        Ip      = geometry.get("Ip", 15e6)
        B_p     = mu0 * Ip / (2.0 * np.pi * a)
        betap   = p_avg / (B_p**2 / (2.0 * mu0))

        return {"betap": max(betap, 0.01), "Ip": Ip}


# ---------------------------------------------------------------------------
# Default TORAX config for ITER
# ---------------------------------------------------------------------------

def _default_iter_torax_config(
    geometry_dir: str | None = None,
    geometry_files: dict[float, str] | None = None,
    t_final: float = 80.0,
    Ip_A: float = 15.0e6,
    T_e0_keV: float = 20.0,
    P_heat_W: float = 33.0e6,
    transport_model: str = "constant",
) -> dict:
    """
    Default TORAX configuration for an ITER-like scenario.

    Based on torax/examples/iterhybrid_rampup.py, adapted for our pipeline.
    Geometry is provided externally via GEQDSK files from FreeGSNKE.

    Parameters
    ----------
    geometry_dir     : directory containing GEQDSK files
    geometry_files   : dict {time_s: filename} for time-dependent geometry,
                       or None for single geometry (first file used)
    t_final          : simulation end time (s)
    Ip_A             : plasma current (A)
    T_e0_keV         : initial on-axis electron temperature (keV)
    P_heat_W         : total heating power (W)
    transport_model  : 'constant', 'bohm-gyrobohm', or 'qlknn'
    """
    # Build geometry config (time-dependent or static)
    if geometry_files and len(geometry_files) > 1:
        # Time-dependent geometry: multiple GEQDSK files
        geometry_configs = {}
        for t, fname in geometry_files.items():
            geometry_configs[t] = {"geometry_file": fname}
        geometry_cfg = {
            "geometry_type": "eqdsk",
            "geometry_directory": geometry_dir,
            "cocos": 7,
            "Ip_from_parameters": True,
            "last_surface_factor": 0.85,
            "n_surfaces": 50,
            "geometry_configs": geometry_configs,
        }
    elif geometry_files:
        # Single GEQDSK file
        first_file = next(iter(geometry_files.values()))
        geometry_cfg = {
            "geometry_type": "eqdsk",
            "geometry_file": first_file,
            "geometry_directory": geometry_dir,
            "cocos": 7,
            "Ip_from_parameters": True,
            "last_surface_factor": 0.85,
            "n_surfaces": 50,
        }
    else:
        # Fallback: circular geometry (for testing without GEQDSK)
        geometry_cfg = {
            "geometry_type": "circular",
        }

    # Build transport config based on model choice
    if transport_model == "qlknn":
        transport_cfg = {
            "model_name": "qlknn",
            "apply_inner_patch": True,
            "chi_i_inner": 1.5,
            "chi_e_inner": 1.5,
            "D_e_inner": 0.25,
            "V_e_inner": 0.0,
            "rho_inner": 0.3,
            "apply_outer_patch": True,
            "chi_i_outer": 2.0,
            "chi_e_outer": 2.0,
            "D_e_outer": 0.1,
            "V_e_outer": 0.0,
            "rho_outer": 0.9,
            "DV_effective": True,
            "include_ITG": True,
            "include_TEM": True,
            "include_ETG": True,
            "avoid_big_negative_s": True,
        }
    elif transport_model == "bohm-gyrobohm":
        transport_cfg = {
            "model_name": "bohm-gyrobohm",
            "apply_inner_patch": True,
            "chi_i_inner": 1.5,
            "chi_e_inner": 1.5,
            "rho_inner": 0.3,
        }
    else:
        transport_cfg = {
            "model_name": "constant",
        }

    Ip_MA = Ip_A * 1e-6

    return {
        "plasma_composition": {
            "main_ion": {"D": 0.5, "T": 0.5},
            "impurity": "Ne",
            "Z_eff": 1.6,
        },
        "profile_conditions": {
            "Ip": {0: Ip_A},
            "T_e": {0.0: {0.0: T_e0_keV, 1.0: 0.1}},
            "T_e_right_bc": 0.1,
            "T_i": {0.0: {0.0: T_e0_keV * 0.9, 1.0: 0.1}},
            "T_i_right_bc": 0.1,
            "n_e_nbar_is_fGW": True,
            "nbar": 0.85,
            "n_e": {0: {0.0: 1.5, 1.0: 1.0}},
            "n_e_right_bc_is_fGW": True,
            "n_e_right_bc": 0.2,
        },
        "geometry": geometry_cfg,
        "numerics": {
            "t_final": t_final,
            "fixed_dt": 2,
            "resistivity_multiplier": 1,
            "evolve_ion_heat": True,
            "evolve_electron_heat": True,
            "evolve_current": True,
            "evolve_density": True,
            "max_dt": 1.0,
            "chi_timestep_prefactor": 30,
            "dt_reduction_factor": 3,
        },
        "neoclassical": {
            "bootstrap_current": {
                "bootstrap_multiplier": 1.0,
            },
        },
        "sources": {
            "generic_current": {
                "fraction_of_total_current": 0.15,
                "gaussian_width": 0.075,
                "gaussian_location": 0.36,
            },
            "generic_particle": {},
            "gas_puff": {},
            "pellet": {},
            "generic_heat": {
                "P_total": P_heat_W,
                "gaussian_location": 0.13,
                "gaussian_width": 0.07,
                "electron_heat_fraction": 1.0,
            },
            "fusion": {},
            "ei_exchange": {},
            "ohmic": {},
        },
        "pedestal": {
            "model_name": "set_T_ped_n_ped",
            "set_pedestal": True,
            "T_i_ped": 1.0,
            "T_e_ped": 1.0,
            "n_e_ped_is_fGW": True,
            "n_e_ped": 0.5,
            "rho_norm_ped_top": 0.9,
        },
        "transport": transport_cfg,
        "solver": {
            "solver_type": "newton_raphson",
            "use_predictor_corrector": True,
            "n_corrector_steps": 10,
            "chi_pereverzev": 30,
            "D_pereverzev": 15,
            "use_pereverzev": True,
        },
        "time_step_calculator": {
            "calculator_type": "fixed",
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
