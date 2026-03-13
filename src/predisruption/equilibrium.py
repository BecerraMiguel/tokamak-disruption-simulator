"""
FreeGSNKE equilibrium solver wrapper.

Provides a clean interface to:
  - Static forward solve (for initial condition)
  - Static inverse solve (constrain Ip, betap, shape)
  - Nonlinear dynamic solve (time evolution with circuit equations)

The equilibrium gives: psi(R,Z), q(rho), Ip, betaN, q95, separatrix,
coil currents — everything the transport code needs as geometry input.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np


def _patch_freegs4e_inside_mask():
    """
    Monkey-patch freegs4e.critical.inside_mask to fix a single-X-point crash.

    Bug in freegs4e ≤ 0.x: line 864 reads
        if len(xpoint > 1):
    which evaluates len(bool_array_with_nrows) — for a single X-point this is
    len(array_of_shape_(1,3)) = 1, which is truthy, and the next line then
    tries to access xpoint[1, 2] → IndexError.

    The correct condition is:
        if len(xpoint) > 1:

    This is only triggered when use_geom=True (the default) and the equilibrium
    has exactly one X-point (single-null divertor like ITER 15 MA flat-top).
    """
    try:
        import numpy as np
        import freegs4e.critical as _crit

        _inner  = _crit.inside_mask_
        _geom   = _crit.geom_inside_mask

        def _patched(
            R, Z, psi, opoint, xpoint=[], mask_outside_limiter=None,
            psi_bndry=None, use_geom=True,
        ):
            mask = _inner(R, Z, psi, opoint, xpoint, mask_outside_limiter, psi_bndry)
            if use_geom:
                mask = mask * _geom(R, Z, opoint, xpoint)
                # FIX: was `len(xpoint > 1)` which equals len(bool_array) == nrows
                # and is truthy even for a single X-point → IndexError on xpoint[1,2].
                if len(xpoint) > 1:
                    if (
                        np.abs(
                            (xpoint[0, 2] - xpoint[1, 2])
                            / (opoint[0, 2] - xpoint[0, 2])
                        ) < 0.1
                    ):
                        mask = mask * _geom(R, Z, opoint, xpoint[1:])
            return mask

        _crit.inside_mask = _patched
    except Exception:
        pass   # freegs4e not installed → skip silently


def _patch_freegsnke_copy_into():
    """
    Monkey-patch FreeGSNKE's copy_into() to handle None values.

    Bug in FreeGSNKE 2.1.0: when diverted_core_mask (or similar mutable attrs)
    is None (limiter-bounded equilibrium, no X-point), profiles.copy() raises
      TypeError: Cannot copy <class 'NoneType'> without deepcopying
    because copy_into() tries to np.copy(None) and fails.

    Fix: when the attribute value is None, just assign None directly (it's
    immutable so no copy is needed).
    """
    try:
        import freegsnke.copying as _fc
        import freegsnke.jtor_update as _jt

        _orig = _fc.copy_into

        def _patched(obj, new_obj, attr, *, mutable=False, strict=True,
                     allow_deepcopy=False):
            if not hasattr(obj, attr) and not strict:
                return
            attribute_value = getattr(obj, attr)
            if mutable and attribute_value is None:
                # None is immutable; assign directly without copying.
                setattr(new_obj, attr, None)
                return
            return _orig(obj, new_obj, attr, mutable=mutable, strict=strict,
                         allow_deepcopy=allow_deepcopy)

        _fc.copy_into = _patched
        _jt.copy_into = _patched   # patch the imported reference too
    except Exception:
        pass   # if FreeGSNKE isn't installed, skip silently


_patch_freegs4e_inside_mask()
_patch_freegsnke_copy_into()


class EquilibriumSolver:
    """
    Wrapper around FreeGSNKE NKGSsolver and nl_solver.

    Parameters
    ----------
    tokamak : FreeGSNKE Machine object
        Built by build_iter_machine().
    nx, ny : int
        Grid resolution. Must be 2^n+1. Use 65 for fast runs, 129 for accuracy.
    Rmin, Rmax, Zmin, Zmax : float
        Computational domain (m). Defaults to ITER domain from config.
    verbose : bool
        Print solver progress.
    """

    def __init__(
        self,
        tokamak,
        nx: int = 65,
        ny: int = 65,
        Rmin: float = 3.0,
        Rmax: float = 9.0,
        Zmin: float = -6.0,
        Zmax: float = 6.0,
        verbose: bool = False,
    ):
        self.tokamak = tokamak
        self.nx = nx
        self.ny = ny
        self.domain = (Rmin, Rmax, Zmin, Zmax)
        self.verbose = verbose

        self._static_solver = None
        self._dynamic_solver = None
        self._eq = None            # current equilibrium object
        self._profiles = None      # current profile object

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _get_eq_object(self):
        """Create or return the FreeGSNKE Equilibrium object (defines the grid).

        Uses freegsnke.equilibrium_update.Equilibrium (not freegs4e.Equilibrium)
        because it adds limiter_handler and other FreeGSNKE-specific attributes
        needed by the profile and solver classes.
        """
        if self._eq is None:
            from freegsnke.equilibrium_update import Equilibrium
            import freegs4e
            Rmin, Rmax, Zmin, Zmax = self.domain
            self._eq = Equilibrium(
                tokamak=self.tokamak,
                Rmin=Rmin, Rmax=Rmax,
                Zmin=Zmin, Zmax=Zmax,
                nx=self.nx, ny=self.ny,
                boundary=freegs4e.boundary.freeBoundary,
            )
        return self._eq

    def _get_static_solver(self):
        if self._static_solver is None:
            from freegsnke.GSstaticsolver import NKGSsolver
            eq = self._get_eq_object()
            self._static_solver = NKGSsolver(eq)
        return self._static_solver

    def _make_profiles(self, betap: float, Ip: float,
                        alpha_m: float = 1.0, alpha_n: float = 2.0):
        """Create a ConstrainBetapIp profile object."""
        from freegsnke.jtor_update import ConstrainBetapIp
        from .iter_machine import ITER_PARAMS
        fvac = ITER_PARAMS["fvac"]
        Raxis = ITER_PARAMS["R_major"]
        eq = self._get_eq_object()
        return ConstrainBetapIp(eq, betap=betap, Ip=Ip, fvac=fvac,
                                alpha_m=alpha_m, alpha_n=alpha_n,
                                Raxis=Raxis)

    # ------------------------------------------------------------------
    # Static solve (initial condition)
    # ------------------------------------------------------------------

    def solve_static(
        self,
        Ip: float,
        betap: float = 0.5,
        coil_targets: dict | None = None,
        xpoints: list | None = None,
        isoflux: list | None = None,
    ):
        """
        Compute a static MHD equilibrium using inverse_solve.

        Uses FreeGSNKE's inverse solver with null-point constraints
        (O-point at R=6.2m, X-point at R=6.0m, Z=-3.8m) and isoflux
        constraints (X-point + top-of-plasma) to produce an ITER-like
        lower single-null equilibrium with kappa~1.75, delta~0.33.

        Parameters
        ----------
        Ip : float
            Target plasma current (A).
        betap : float
            Poloidal beta (dimensionless). ~0.5 for ITER 15 MA flat-top.
        coil_targets : dict, optional
            {coil_name: current_A} — initial coil currents.
            If None, uses reference_coil_currents_15MA().
        xpoints : list, optional
            Reserved for future use.
        isoflux : list, optional
            Reserved for future use.

        Returns
        -------
        eq : FreeGSNKE equilibrium object
        """
        solver = self._get_static_solver()
        profiles = self._make_profiles(betap=betap, Ip=Ip)
        eq = self._get_eq_object()

        # Apply coil currents.  Default to ITER 15 MA flat-top reference if
        # not supplied.  _apply_coil_currents syncs both circuit.current AND
        # tokamak.current_vec so that eq.psi() = plasma_psi + tokamak_psi
        # uses the correct external coil field.
        if coil_targets is None:
            from .iter_machine import reference_coil_currents_15MA
            coil_targets = reference_coil_currents_15MA()
        self._apply_coil_currents(coil_targets)

        # Initialise plasma_psi on the first call only.
        #
        # A Gaussian centred at (R=6.2, Z=0) — the ITER magnetic axis — is
        # essential.  Using adaptive_centre=True places the Gaussian at the
        # limiter centroid (~Z=-2m for ITER's bottom-divertor limiter), which
        # pulls the axis toward the X-point.  The widths (σ_R, σ_Z) are chosen
        # to roughly match ITER's minor radius and elongated shape.
        #
        # The amplitude must exceed ~5.4 Wb to create an O-point against the
        # ITER coil field (coil psi at axis ≈ −5.9 Wb).  We scale to 80×
        # the default peak for margin.
        if not eq.solved:
            R, Z = eq.R, eq.Z
            sigma_R, sigma_Z = 1.5, 2.5  # m
            gauss = np.exp(-((R - 6.2)**2 / (2 * sigma_R**2)
                             + (Z - 0.0)**2 / (2 * sigma_Z**2)))
            ref_peak = eq.create_psi_plasma_default(adaptive_centre=True).max()
            eq.plasma_psi = gauss * (80.0 * ref_peak / gauss.max())

        # Use FreeGSNKE's inverse_solve (Picard + NK with shape constraints).
        #
        # forward_solve converges to R≈7.9m (wrong axis) because the ITER
        # coil psi is nearly flat across the midplane — there is no
        # topological attractor at R=6.2m from the coil field alone.
        #
        # inverse_solve uses an Inverse_optimizer to constrain:
        #   - null_points: X-point at (6.0, -3.8) and O-point at (6.2, 0.0)
        #   - isoflux_set: boundary points on the same flux surface to enforce
        #     the correct plasma shape (kappa ≈ 1.75, delta ≈ 0.33)
        #   - force_up_down_symmetric: keeps Z_axis ≈ 0
        from freegsnke.inverse import Inverse_optimizer

        # ITER lower single-null shape parameters
        R0, a_min = 6.2, 2.0     # major radius, minor radius (m)
        kappa_t, delta_t = 1.75, 0.33  # target elongation, triangularity
        R_xpt, Z_xpt = 6.0, -3.8      # lower X-point
        R_out = R0 + a_min             # 8.2 m, outboard midplane
        R_top = R0 - delta_t * a_min   # 5.54 m, top of plasma
        Z_top = kappa_t * a_min        # 3.5 m, top of plasma

        constrain = Inverse_optimizer(
            null_points=[[R_xpt, R0], [Z_xpt, 0.0]],
            isoflux_set=[[[R_xpt, R_top], [Z_xpt, Z_top]]],
        )
        constrain.prepare_for_solve(eq)

        # Use tight tolerance so the solver doesn't "converge" prematurely
        # at a topology-jumped state (R~7.9m).  With 1e-4 tolerance the
        # solver never declares success (the relative error oscillates at
        # ~2-5e-2), but after 15 iterations the equilibrium is stable with
        # R~6.2m, kappa~1.78, delta~0.32.  Beyond ~25 iterations the
        # forward_solve sub-step may jump to a wrong equilibrium basin.
        converged = True
        try:
            solver.inverse_solve(
                eq, profiles, constrain,
                target_relative_tolerance=1e-4,
                max_solving_iterations=15,
                verbose=self.verbose,
                suppress=not self.verbose,
                force_up_down_symmetric=True,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[EquilibriumSolver] inverse_solve error: {exc}")
            converged = False

        if self.verbose:
            final_err = getattr(solver, "best_relative_change", float("inf"))
            status = (
                f"CONVERGED (err={final_err:.3e})" if converged
                else f"partial (err={final_err:.3e})"
            )
            print(f"[EquilibriumSolver] static solve {status}.")

        # NOTE: Do NOT restore solver.best_psi here.  The solver's "best"
        # iteration (lowest relative_change) often corresponds to an
        # equilibrium with the wrong magnetic axis (R~7.8m) because the
        # convergence metric doesn't account for topology.  The final
        # iteration's plasma_psi — which is what inverse_solve leaves on
        # eq — has the correct axis position (R~6.2m) when the null-point
        # constraints are active.

        # Populate FreeGSNKE-specific attributes (xpt, opt, psi_bndry, …)
        # that the dynamic solver and signal extractor expect.
        try:
            solver.port_critical(eq, profiles)
        except Exception as exc:
            if self.verbose:
                print(f"[EquilibriumSolver] port_critical failed: {exc}")
            eq.solved = True   # mark done anyway so subsequent calls don't re-init

        self._profiles = profiles
        return eq

    # ------------------------------------------------------------------
    # Dynamic solve (time evolution)
    # ------------------------------------------------------------------

    def init_dynamic(
        self,
        eq_init,
        profiles_init=None,
        betap: float = 0.5,
        Ip: float = 15.0e6,
    ):
        """
        Initialise the nonlinear dynamic solver from a static equilibrium.

        Parameters
        ----------
        eq_init : FreeGSNKE equilibrium (from solve_static)
        profiles_init : profile object, optional (uses betap/Ip if None)
        betap, Ip : used if profiles_init is None
        """
        from freegsnke.nonlinear_solve import nl_solver

        if profiles_init is None:
            profiles_init = self._make_profiles(betap=betap, Ip=Ip)

        static_solver = self._get_static_solver()

        self._dynamic_solver = nl_solver(
            profiles=profiles_init,
            eq=eq_init,
            GSStaticSolver=static_solver,
            verbose=self.verbose,
        )
        self._dynamic_solver.initialize_from_ICs(eq_init, profiles_init)
        self._profiles = profiles_init

    def step(self, dt: float, V_coils: dict, betap: float, Ip: float):
        """
        Advance the equilibrium by one time step.

        Parameters
        ----------
        dt : float
            Time step (s).
        V_coils : dict
            {coil_name: voltage_V} applied to each active circuit.
        betap : float
            Updated poloidal beta (from transport solver).
        Ip : float
            Updated plasma current (from transport solver).

        Returns
        -------
        eq : updated equilibrium object
        """
        if self._dynamic_solver is None:
            raise RuntimeError("Call init_dynamic() before step().")

        # Build voltage array in coil order
        V_array = self._coil_voltages_to_array(V_coils)

        # Update profile parameters and time step size
        profiles = self._make_profiles(betap=betap, Ip=Ip)
        self._dynamic_solver.dt_step = dt

        self._dynamic_solver.nlstepper(V_array)
        self._eq = self._dynamic_solver.eq1   # updated equilibrium
        self._profiles = profiles
        return self._eq

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------

    def get_signals(self, eq=None, n_rho: int = 51) -> dict:
        """
        Extract all physics signals from the current equilibrium.

        Parameters
        ----------
        eq : equilibrium object (uses self._eq if None)
        n_rho : int
            Number of radial grid points for profiles.

        Returns
        -------
        signals : dict with keys:
            "Ip"          : float, total plasma current (A)
            "betaN"       : float, normalised beta
            "betap"       : float, poloidal beta
            "q95"         : float, safety factor at 95% flux surface
            "q_min"       : float, minimum safety factor
            "li"          : float, internal inductance
            "kappa"       : float, elongation
            "delta"       : float, triangularity
            "q_profile"   : ndarray [n_rho], q(rho) on uniform rho grid
            "psi_profile" : ndarray [n_rho], psi(rho_N) (normalised flux)
            "rho"         : ndarray [n_rho], normalised radial coordinate
            "R_sep", "Z_sep" : ndarrays, separatrix (R,Z) points
            "R_axis", "Z_axis" : floats, magnetic axis
            "coil_currents" : dict {name: current_A}
        """
        if eq is None:
            eq = self._eq
        if eq is None:
            raise RuntimeError("No equilibrium available — call solve_static() first.")

        rho = np.linspace(0, 1, n_rho)
        rho_inner = rho[1:-1]   # avoid axis singularity and separatrix

        # Safety factor profile (skip axis and LCFS where q diverges)
        q_profile = np.full(n_rho, np.nan)
        try:
            q_profile[1:-1] = eq.q(rho_inner)
            q_profile[0]    = q_profile[1]    # extend to axis
            q_profile[-1]   = q_profile[-2]   # extend to LCFS
        except Exception:
            pass

        # Global scalars
        Ip = betap = li = np.nan
        try:
            Ip = float(eq.plasmaCurrent())
        except Exception:
            pass
        try:
            betap = float(eq.poloidalBeta1())
        except Exception:
            pass
        try:
            li = float(eq.internalInductance1())
        except Exception:
            pass

        # Separatrix: freegs4e returns (ntheta, 2) array of (R, Z) points
        # Computed before betaN because we need a_actual from the separatrix.
        R_sep = Z_sep = np.array([])
        kappa = delta = np.nan
        try:
            sep = eq.separatrix()          # shape (ntheta, 2)
            R_sep = sep[:, 0]
            Z_sep = sep[:, 1]
            if len(R_sep) > 0:
                kappa = (Z_sep.max() - Z_sep.min()) / (R_sep.max() - R_sep.min())
        except Exception:
            pass
        try:
            delta = float(eq.triangularity())
        except Exception:
            pass

        # βN = βt(%) × a × B0 / Ip[MA]  where βt = βp × (μ₀Ip/(2πaB₀))²
        # Simplifies to: βN = 4 × βp × Ip[MA] / (a × B₀)
        # Use actual minor radius from separatrix, not reference value.
        try:
            from .iter_machine import ITER_PARAMS
            if len(R_sep) > 0:
                a_actual = (R_sep.max() - R_sep.min()) / 2.0
            else:
                a_actual = ITER_PARAMS["a_minor"]  # fallback
            B0 = ITER_PARAMS["B0"]
            betaN = 4.0 * betap * abs(Ip) / (1e6 * a_actual * B0)
        except Exception:
            betaN = np.nan

        q95   = float(q_profile[int(0.95 * n_rho)]) if not np.isnan(q_profile).all() else np.nan
        q_min = float(np.nanmin(q_profile))

        # Psi profile (normalised)
        psi_profile = np.linspace(0, 1, n_rho)   # ψ_N = ρ²_pol by definition

        # Magnetic axis: freegs4e returns [R, Z, psi]
        R_axis = Z_axis = np.nan
        try:
            axis = eq.magneticAxis()
            R_axis = float(axis[0])
            Z_axis = float(axis[1])
        except Exception:
            pass

        # Minor radius from separatrix
        from .iter_machine import ITER_PARAMS as _ip
        a_minor = (R_sep.max() - R_sep.min()) / 2.0 if len(R_sep) > 0 else _ip["a_minor"]

        # Coil currents
        coil_currents = self._get_coil_currents(eq)

        return {
            "Ip":           Ip,
            "betaN":        betaN,
            "betap":        betap,
            "q95":          q95,
            "q_min":        q_min,
            "li":           li,
            "kappa":        kappa,
            "delta":        delta,
            "a_minor":      a_minor,
            "q_profile":    q_profile,
            "psi_profile":  psi_profile,
            "rho":          rho,
            "R_sep":        R_sep,
            "Z_sep":        Z_sep,
            "R_axis":       R_axis,
            "Z_axis":       Z_axis,
            "coil_currents": coil_currents,
        }

    # ------------------------------------------------------------------
    # GEQDSK I/O (for TORAX geometry coupling)
    # ------------------------------------------------------------------

    def write_geqdsk(self, path: str, eq=None):
        """Write current equilibrium to GEQDSK format for TORAX."""
        if eq is None:
            eq = self._eq
        from freegs4e import geqdsk
        with open(path, "w") as f:
            geqdsk.write(eq, f)

    def write_geqdsk_tmp(self, eq=None) -> str:
        """
        Write GEQDSK to a named temp file and return the path.
        Caller is responsible for deleting the file.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".geqdsk", delete=False, mode="w")
        path = tmp.name
        tmp.close()
        self.write_geqdsk(path, eq)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_coil_currents(self, coil_targets: dict):
        """Set coil currents on the tokamak object.

        tokamak.coils is a list of (name, Circuit) tuples; use coil_order
        for O(1) lookup.

        IMPORTANT: must update BOTH circuit.current AND tokamak.current_vec.
        - circuit.current is used by calcPsiFromGreens (e.g. adjust_psi_plasma)
        - current_vec is used by getPsitokamak (eq.psi() in freegs4e.solve)
        Keeping them in sync ensures both paths see the correct coil flux.
        """
        coil_order = getattr(self.tokamak, "coil_order", {})
        current_vec = getattr(self.tokamak, "current_vec", None)
        for name, current in coil_targets.items():
            if name in coil_order:
                idx = coil_order[name]
                self.tokamak.coils[idx][1].current = float(current)
                # Keep current_vec in sync so getPsitokamak gives correct psi.
                if current_vec is not None and idx < len(current_vec):
                    current_vec[idx] = float(current)

    def _coil_voltages_to_array(self, V_coils: dict) -> np.ndarray:
        """Convert {name: voltage} dict to ordered numpy array for nl_solver."""
        coil_names = list(self.tokamak.coils_list)[:self.tokamak.n_active_coils]
        V = np.zeros(len(coil_names))
        for i, name in enumerate(coil_names):
            if name in V_coils:
                V[i] = V_coils[name]
        return V

    def _get_coil_currents(self, eq) -> dict:
        """Extract current from each active coil circuit."""
        currents = {}
        try:
            coil_order = getattr(self.tokamak, "coil_order", {})
            for name in self.tokamak.coils_list[:self.tokamak.n_active_coils]:
                if name in coil_order:
                    idx = coil_order[name]
                    circuit = self.tokamak.coils[idx][1]
                    currents[name] = getattr(circuit, "current", np.nan)
        except Exception:
            pass
        return currents
