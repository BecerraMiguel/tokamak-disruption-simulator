# Physics Codes: Deep Analysis & Coupling Plan

This document provides a comprehensive analysis of FreeGS, FreeGSNKE, and TORAX, and details how to couple FreeGSNKE + TORAX for our tokamak disruption simulator.

---

## Table of Contents

1. [FreeGS — Static Grad-Shafranov Solver](#1-freegs)
2. [FreeGSNKE — Time-Evolving Equilibrium Solver](#2-freegsnke)
3. [TORAX — 1D Transport Code](#3-torax)
4. [Coupling Analysis: FreeGSNKE + TORAX](#4-coupling-analysis)
5. [Implementation Plan](#5-implementation-plan)

---

## 1. FreeGS

**Repository:** https://github.com/freegs-plasma/freegs
**Install:** `pip install freegs`
**License:** LGPL
**Language:** Pure Python (NumPy, SciPy)

### 1.1 What It Solves

FreeGS solves the **Grad-Shafranov (GS) equation**, the fundamental MHD equilibrium equation for axisymmetric tokamak plasmas:

```
Δ*ψ = -μ₀ R² dp/dψ - F dF/dψ = -μ₀ R J_φ
```

where:
- `Δ* = R ∂/∂R (1/R ∂/∂R) + ∂²/∂Z²` is the Grad-Shafranov operator
- `ψ(R,Z)` is the poloidal magnetic flux function
- `p(ψ)` is the plasma pressure (free function)
- `F(ψ) = R·B_toroidal` is the poloidal current function (free function)
- `J_φ` is the toroidal current density
- `(R,Z,φ)` are cylindrical coordinates with toroidal symmetry in φ

**Physical assumptions:**
- Axisymmetry (∂/∂φ = 0 for all quantities)
- Ideal MHD force balance: ∇p = J × B
- Static equilibrium (no time evolution, no flows)
- Isotropic pressure

**What the solution gives:**
- ψ(R,Z): poloidal flux map on 2D grid
- Plasma boundary (last closed flux surface / separatrix)
- X-point and O-point locations
- q(ψ): safety factor profile
- p(ψ): pressure profile
- F(ψ): poloidal current function
- J_tor(R,Z): toroidal current density
- B_R, B_Z, B_tor: magnetic field components
- Ip: total plasma current
- β_p, β_t, β_N: beta values
- l_i: internal inductance
- Shafranov shift, elongation, triangularity

### 1.2 Numerical Methods

**Spatial discretization:** Finite differences on a rectangular (R,Z) grid
- 2nd-order and 4th-order stencils available for the GS operator
- Grid must be (2^n + 1) × (2^n + 1), typically 65×65

**Solver:** Picard iteration (fixed-point iteration)
```python
while not converged:
    # 1. Compute J_tor from current ψ using profile functions
    # 2. Solve linear elliptic PDE: Δ*ψ_new = -μ₀ R J_tor
    # 3. Update boundary conditions
    # 4. Blend: ψ = (1-α)ψ_new + α·ψ_old
    # 5. Check convergence: |ψ_new - ψ_old| < tol
```

Convergence criteria: `max|Δψ| < atol` OR `max|Δψ|/max|ψ| < rtol` (default rtol=1e-3)

**Boundary conditions:**
- `fixedBoundary`: ψ=0 on domain edges (simplest)
- `freeBoundary`: Green's function integration over plasma current for each boundary point
- `freeBoundaryHagenow`: optimized von Hagenow method (faster)

**Linear solver:** Multigrid elliptic solver (V-cycle)

### 1.3 Machine Definitions

FreeGS includes predefined machines:
- `MAST()`, `MAST_sym()`, `MASTU()`, `MASTU_simple()` — Mega-Amp Spherical Tokamak
- `DIIID()` — DIII-D (18 PF coils)
- `TCV()` — TCV with solenoid
- `TestTokamak()` — simple test configuration

**Custom machines** are defined by specifying:
```python
coils = [
    ("CS", Solenoid(Rs=0.15, Zsmin=-1.4, Zsmax=1.4, Ns=100)),
    ("PF1", Coil(R=1.75, Z=0.6, current=0.0, control=True)),
    ("PF2", Circuit([
        ("PF2U", Coil(0.49, 1.76), 1.0),
        ("PF2L", Coil(0.49, -1.76), 1.0)
    ])),
]
wall = Wall(R_array, Z_array)
machine = Machine(coils, wall=wall)
```

Coil types: `Coil` (single filament), `ShapedCoil` (polygon), `FilamentCoil` (multi-filament), `Solenoid`, `Circuit` (linked coils), `MirroredCoil` (up-down symmetric pair).

### 1.4 Plasma Profiles

FreeGS provides parametric profile classes:
- **`ConstrainPaxisIp(paxis, Ip, fvac)`**: Specify axis pressure and plasma current → derives p'(ψ) and ff'(ψ)
- **`ConstrainBetapIp(betap, Ip, fvac)`**: Specify poloidal beta and plasma current
- **`ProfilesPprimeFfprime(pprime_func, ffprime_func, fvac)`**: User-supplied arbitrary profile functions

Profile shape uses Lao parameterization:
```
J_tor = L · (β₀·R/R_axis + (1-β₀)·R_axis/R) · (1 - ψ_N^α_m)^α_n
```
where α_m, α_n control the profile peaking/broadness.

### 1.5 Coil Current Control

The `control.py` module optimizes coil currents to satisfy constraints:
- **xpoints**: Force X-point at specified (R,Z) location
- **isoflux**: Force equal ψ at pairs of points (shapes the plasma boundary)
- **psivals**: Set ψ to specific values at locations
- **current_lims**: Coil current bounds

Optimization uses Tikhonov regularization + SLSQP constrained optimization.

### 1.6 API Usage Example

```python
import freegs

# 1. Create machine
tokamak = freegs.machine.TestTokamak()

# 2. Create equilibrium on grid
eq = freegs.Equilibrium(tokamak=tokamak,
                        Rmin=0.1, Rmax=2.0,
                        Zmin=-1.0, Zmax=1.0,
                        nx=65, ny=65,
                        boundary=freegs.boundary.freeBoundaryHagenow)

# 3. Define profiles
profiles = freegs.jtor.ConstrainPaxisIp(
    paxis=1e3,    # Pa, pressure on axis
    Ip=1e6,       # A, plasma current
    fvac=0.5      # f = R*Bt in vacuum
)

# 4. Define shape constraints
xpt = [(1.1, -0.6)]  # X-point location
isoflux = [(1.1, -0.6, 1.1, 0.6)]  # up-down symmetric
control = freegs.control.constrain(xpoints=xpt, isoflux=isoflux)

# 5. Solve
freegs.solve(eq, profiles, constrain=control, rtol=1e-3, maxits=50)

# 6. Access results
print(f"Ip = {eq.plasmaCurrent()} A")
print(f"beta_N = {eq.betaN()}")
print(f"q95 = {eq.q(0.95)}")
R_sep, Z_sep = eq.separatrix()
```

### 1.7 Outputs Available from Equilibrium Object

| Method | Returns | Units |
|--------|---------|-------|
| `eq.psi()` / `eq.psiRZ(R,Z)` | Poloidal flux ψ(R,Z) | Wb |
| `eq.q(psinorm)` | Safety factor profile | - |
| `eq.pressure(psinorm)` | Pressure profile | Pa |
| `eq.fpol(psinorm)` | F = R·Bt profile | T·m |
| `eq.pprime(psinorm)` | dp/dψ | Pa/Wb |
| `eq.ffprime(psinorm)` | F·dF/dψ | T²·m²/Wb |
| `eq.plasmaCurrent()` | Total Ip | A |
| `eq.poloidalBeta()` | β_p | - |
| `eq.toroidalBeta()` | β_t | - |
| `eq.betaN()` | Normalized beta | - |
| `eq.internalInductance()` | l_i | - |
| `eq.elongation()` | κ | - |
| `eq.triangularity()` | δ | - |
| `eq.separatrix(npoints)` | (R,Z) of LCFS | m |
| `eq.magneticAxis()` | (R,Z) of O-point | m |
| `eq.Br(R,Z)`, `eq.Bz(R,Z)` | Magnetic field | T |
| `eq.plasmaVolume()` | Plasma volume | m³ |
| `eq.w_th()` | Thermal stored energy | J |

### 1.8 I/O Formats

FreeGS can read/write **GEQDSK** (G-EQDSK) files — the standard equilibrium interchange format:
```python
from freegs import geqdsk
with open("equilibrium.geqdsk", "w") as f:
    geqdsk.write(eq, f)
```

### 1.9 Limitations

- **Static only**: No time evolution — computes one equilibrium at a time
- **No transport**: Cannot evolve Te, ne, j profiles — these must come from external transport code
- **Picard solver**: Slower convergence than Newton methods; can struggle with high-beta or strongly-shaped equilibria
- **No passive structures**: Cannot model eddy currents in vessel walls
- **No circuit equations**: Coil currents are specified directly, not evolved via circuit equations

### 1.10 Relationship to FreeGSNKE

FreeGS is the **foundation** that FreeGSNKE extends. FreeGSNKE (through its dependency FreeGS4E, a fork of FreeGS) uses FreeGS's grid, machine definitions, and profile structures, but replaces the Picard solver with Newton-Krylov and adds time evolution + circuit equations + passive structures.

---

## 2. FreeGSNKE

**Repository:** https://github.com/FusionComputingLab/freegsnke
**Documentation:** https://docs.freegsnke.com/
**Install:** `pip install "freegsnke[freegs4e]"`
**License:** LGPL v3
**Paper:** Amorisco et al., "FreeGSNKE: A Python-based dynamic free-boundary toroidal plasma equilibrium solver", *Physics of Plasmas*, **31**, 042517 (2024). DOI: 10.1063/5.0188467
**Validation:** Pentland et al. (2025), validated against EFIT++ on MAST-U, DOI: 10.1088/1402-4896/ada192

### 2.1 What FreeGSNKE Adds Over FreeGS

FreeGSNKE introduces three major capabilities:

1. **Newton-Krylov static solver**: Replaces Picard iteration with a Newton-Krylov (NK) method for the static GS equation. Uses Arnoldi iteration for the Krylov subspace. 4th-order accurate finite differences. Better stability and convergence, especially for challenging equilibria.

2. **Linearized dynamic solver**: Solves the linearized time-dependent equilibrium problem — used for stability analysis (vertical stability margin, growth rates).

3. **Nonlinear dynamic solver**: Full time evolution of the free-boundary equilibrium coupled with circuit equations for PF coils and passive structures (eddy currents). This is the key feature for our project.

### 2.2 Physics of Time Evolution

FreeGSNKE evolves the coupled system:

**Grad-Shafranov equation** (at each time step):
```
Δ*ψ = -μ₀ R J_tor(R,Z,t)
```

**Circuit equations** for active coils and passive structures:
```
L · dI/dt + R · I = V_applied(t) + V_induced(t)
```

where:
- `L` is the mutual inductance matrix (coils + passive structures + plasma)
- `R` is the resistance matrix
- `V_applied(t)` are the externally applied voltages (user-prescribed)
- `V_induced(t)` includes flux coupling to plasma

The **plasma current density** profile is prescribed via parametric profiles (same Lao parameterization as FreeGS), with parameters that can be time-dependent.

**Implicit Euler time stepping** is used for the circuit equations, coupled with the NK solver for the GS equation at each step.

### 2.3 Key Classes and API

**Core solver classes:**
- `NKGSsolver` — Newton-Krylov Grad-Shafranov solver (static forward/inverse)
- `nl_solver` (nonlinear_solve) — Nonlinear time evolution solver
- `linear_solver` — Linearized evolution (stability analysis)
- `implicit_euler_solver` — Time integration
- `metal_currents` — Circuit equation handler for coils + passive structures

**Machine configuration:**
```python
from freegsnke import build_machine

tokamak = build_machine.tokamak(
    active_coils_path="active_coils.pickle",
    passive_coils_path="passive_coils.pickle",
    limiter_path="limiter.pickle",
    wall_path="wall.pickle",
    magnetic_probe_path="magnetic_probes.pickle",
)
```

Active coils are defined with:
- `R, Z`: position
- `dR, dZ`: dimensions
- `resistivity`: (e.g., copper = 1.55e-8 Ωm)
- `polarity`, `multiplier`: circuit wiring

Passive structures can be filaments or polygonal shapes (automatically refined into distributed filaments for accurate eddy current modeling). Resistivity for steel: ~5.5e-7 Ωm.

**Profile classes (jtor_update module):**
- `ConstrainBetapIp` — constrain β_p and Ip
- `ConstrainPaxisIp` — constrain axis pressure and Ip
- `Fiesta_Topeol` — alternative parameterization
- `Lao85` — Lao 1985 parameterization
- `TensionSpline` — spline-based profiles
- `Jtor_universal` — unified interface with refinement support

### 2.4 Time Evolution Workflow

Based on the documentation examples (example5 — evolutive forward solve):

```python
# 1. Build machine (coils, passives, limiter, wall)
tokamak = build_machine.tokamak(...)

# 2. Compute initial static equilibrium
static_solver = NKGSsolver(tokamak, ...)
profiles = ConstrainBetapIp(betap=0.5, Ip=1e6, fvac=...)
eq_init = static_solver.forward_solve(profiles)

# 3. Set up nonlinear dynamic solver
nl = nl_solver(tokamak, ...)
nl.initialize_from_ICs(eq_init, profiles)

# 4. Define time-dependent inputs
# - Coil voltages: V_PF(t) for each active coil
# - Profile parameters: betap(t), Ip(t) trajectories

# 5. Time step loop
dt = 0.001  # seconds
for step in range(n_steps):
    nl.nlstepper(dt, V_applied, profile_params)
    # Access current equilibrium state
    eq = nl.equilibrium
    Ip = eq.plasmaCurrent()
    q95 = eq.q(0.95)
    psi = eq.psi()
    coil_currents = nl.get_vessel_currents()
```

### 2.5 Outputs at Each Time Step

From the equilibrium object (same as FreeGS, plus):
- ψ(R,Z) — poloidal flux on 2D grid
- Plasma boundary (separatrix)
- q(ψ_N) — safety factor profile
- Ip — total plasma current
- β_p, β_N — beta values
- κ, δ — elongation, triangularity
- Coil currents (all active + passive)
- Eddy currents in passive structures
- Magnetic probe signals (if configured)

**What FreeGSNKE does NOT provide:**
- Te(ρ,t), Ti(ρ,t) — temperature profiles
- ne(ρ,t) — density profiles
- Transport coefficients (χ, D, V)
- Radiation losses
- Heating deposition profiles
- Bootstrap current

→ These come from TORAX.

### 2.6 Machines with Existing Configurations

FreeGSNKE has examples for:
- **MAST-U** — primary validation target (static + evolutive examples)
- **SPARC** — static inverse example
- **ITER** — static inverse example

For our project, we need to build or adapt an ITER machine configuration.

### 2.7 Limitations

- **No internal transport**: Plasma profiles (p(ψ), J(ψ)) are prescribed via parametric functions, not self-consistently evolved by transport equations
- **Profile parameters must be externally driven**: To have self-consistent evolution, must couple with a transport code
- **Transport coupling is "under development"**: The official FreeGSNKE ↔ transport coupling is not yet released
- **JAX port in progress**: An auto-differentiable JAX version is being developed but not yet available
- **Validated primarily on MAST-U**: ITER-scale validation is limited
- **Python performance**: Can be slow for large grids or long time evolution

---

## 3. TORAX

**Repository:** https://github.com/google-deepmind/torax
**Documentation:** https://torax.readthedocs.io/
**Install:** `pip install torax`
**License:** Apache 2.0
**Paper:** Citrin et al., "TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX", arXiv:2406.06718 (2024)
**Developed by:** Google DeepMind

### 3.1 What TORAX Solves

TORAX solves four coupled 1D transport PDEs on the normalized toroidal flux coordinate ρ_tor (ρ ∈ [0, 1]):

**1. Ion heat transport:**
```
3/2 · ∂(n_i T_i)/∂t = -1/V' · ∂/∂ρ(V' · q_i) + Q_i
```

**2. Electron heat transport:**
```
3/2 · ∂(n_e T_e)/∂t = -1/V' · ∂/∂ρ(V' · q_e) + Q_e
```

**3. Electron particle transport (continuity):**
```
∂n_e/∂t = -1/V' · ∂/∂ρ(V' · Γ_e) + S_e
```

**4. Current diffusion:**
```
∂ψ/∂t = 1/(μ₀σ) · (geometric terms) · ∂²ψ/∂ρ² + ...
```

where:
- V' = ∂V/∂ρ is the volume derivative (from geometry)
- q_i, q_e are heat fluxes (diffusive + convective)
- Q_i, Q_e are heat source/sink terms
- Γ_e is particle flux
- S_e is particle source
- σ is plasma conductivity (Spitzer or Sauter model)

### 3.2 Transport Models

TORAX includes multiple turbulent transport models:

| Model | Description | Speed |
|-------|-------------|-------|
| `constant` | Fixed χ_i, χ_e, D_e, V_e | Fastest |
| `CGM` | Critical gradient model | Fast |
| `bohm-gyrobohm` | Bohm + gyro-Bohm scaling | Fast |
| `qlknn` | **Neural network surrogate for QuaLiKiz** (gyrokinetic) | Fast |
| `tglfnn-ukaea` | Neural network surrogate for TGLF | Fast |
| `qualikiz` | Direct QuaLiKiz call | Slow |
| `combined` | Mix multiple models | Varies |

**QLKNN** is the recommended model — trained on ~300,000 QuaLiKiz gyrokinetic simulations. Predicts:
- χ_i (ion heat diffusivity) from ITG, TEM modes
- χ_e (electron heat diffusivity) from ITG, TEM, ETG modes
- D_e (particle diffusivity)
- V_e (particle convection velocity)

Neoclassical transport: Sauter model for bootstrap current and conductivity; Angioni-Sauter for neoclassical transport.

### 3.3 Source Terms

TORAX includes physics-based models for:

| Source | Physics |
|--------|---------|
| `ohmic` | Ohmic heating: P_OH = η·j² |
| `fusion` | D-T fusion power (α heating) |
| `ecrh` | Electron cyclotron heating (Gaussian deposition) |
| `icrh` | Ion cyclotron heating |
| `bremsstrahlung` | Radiation loss |
| `cyclotron_radiation` | Cyclotron radiation loss |
| `impurity_radiation` | Line radiation from impurities |
| `ei_exchange` | Ion-electron energy exchange (collisional) |
| `neoclassical` | Bootstrap current + Spitzer conductivity |
| `gas_puff` | Particle source (edge fueling) |
| `pellet` | Pellet injection (localized particle source) |
| `generic_heat` | User-defined heat source profile |
| `generic_current` | User-defined current drive profile |
| `generic_particle` | User-defined particle source |

### 3.4 Geometry System (Critical for Coupling)

TORAX needs a **Geometry** object that encodes the flux-surface-averaged magnetic geometry. This is the interface point for coupling with an equilibrium solver.

**Supported geometry types:**
- `circular` — Circular cross-section (analytical, no external file needed)
- `chease` — CHEASE equilibrium file (`.mat2cols` format)
- `eqdsk` — G-EQDSK format (standard equilibrium interchange)
- `fbt` — FBT format (specific to certain codes)
- `imas` — IMAS equilibrium IDS

**What the Geometry object contains:**

The `StandardGeometryIntermediates` dataclass holds (from `standard_geometry.py`):

```
Mesh quantities:
  - rho: normalized toroidal flux coordinate (cell centers)
  - rho_face: face grid locations
  - drho_norm: grid spacing

Flux surface averages (as functions of ρ):
  - <|∇ρ|²>         — g0 coefficient
  - <|∇ρ|²/R²>      — g1 coefficient
  - <1/R²>           — g2 coefficient
  - <B²>             — flux-surface-averaged B²
  - <1/B²>
  - <1/R>

Shape quantities:
  - R_inboard(ρ), R_outboard(ρ)  — midplane radii of each flux surface
  - elongation(ρ), delta_upper(ρ), delta_lower(ρ)  — plasma shape

Flux quantities:
  - psi(ρ)           — poloidal flux
  - Phi(ρ)           — toroidal flux
  - F(ρ) = R·Bt      — toroidal field function
  - Ip_profile(ρ)    — enclosed current profile

Volume/area:
  - V'(ρ) = ∂V/∂ρ   — volume derivative (critical for transport equations)
  - S'(ρ) = ∂A/∂ρ   — surface area derivative

Global parameters:
  - R_major, a_minor  — major/minor radius
  - B_0               — vacuum toroidal field on axis
```

The function `build_standard_geometry()` converts these intermediates into the final `StandardGeometry` object with quantities interpolated to both cell-center and face grids.

**Key insight for coupling:** TORAX already supports **EQDSK geometry** (`geometry_type: 'eqdsk'`). FreeGS/FreeGSNKE can write GEQDSK files. This is the natural coupling interface.

### 3.5 Time-Dependent Geometry

TORAX supports time-varying geometry via `geometry_configs`:
```python
'geometry': {
    'geometry_type': 'eqdsk',
    'geometry_configs': {
        0.0: {'geometry_file': 'eq_t0.geqdsk'},
        1.0: {'geometry_file': 'eq_t1.geqdsk'},
        2.0: {'geometry_file': 'eq_t2.geqdsk'},
    }
}
```

TORAX interpolates between geometry snapshots. This enables loose coupling with an equilibrium solver.

### 3.6 Configuration System

TORAX uses Python dictionaries (not YAML). Example ITER hybrid scenario:

```python
CONFIG = {
    'plasma_composition': {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {'Ne': None},
        'Z_eff': 1.6,
    },
    'geometry': {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
        'R_major': 6.2,    # m
        'a_minor': 2.0,    # m
        'B_0': 5.3,        # T
    },
    'profile_conditions': {
        'Ip': {0: 3e6, 80: 10.5e6},  # A, ramp from 3 to 10.5 MA
        'T_e': {0.0: {0: 6.0, 1: 0.1}},  # keV, core to edge
        'n_e': {0.0: {0: 1.2e20, 1: 0.8e20}},  # m⁻³
    },
    'transport': {
        'model_name': 'qlknn',
        'include_ITG': True,
        'include_TEM': True,
        'include_ETG': True,
    },
    'numerics': {
        't_initial': 0.0,
        't_final': 80.0,
        'fixed_dt': 2.0,
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
        'evolve_current': True,
    },
    'solver': {
        'solver_type': 'newton_raphson',
        'n_corrector_steps': 10,
    },
    'sources': {
        'ecrh': {
            'mode': 'MODEL',
            'P_tot': 20e6,  # 20 MW
        },
        'fusion': {'mode': 'MODEL'},
        'ohmic': {'mode': 'MODEL'},
        'bremsstrahlung': {'mode': 'MODEL'},
        'ei_exchange': {'mode': 'MODEL'},
    },
}
```

### 3.7 Outputs (CoreProfiles State)

At each time step, TORAX produces:

| Field | Description | Units |
|-------|-------------|-------|
| `T_e(ρ)` | Electron temperature | keV |
| `T_i(ρ)` | Ion temperature | keV |
| `n_e(ρ)` | Electron density | m⁻³ |
| `n_i(ρ)` | Main ion density | m⁻³ |
| `psi(ρ)` | Poloidal flux | Wb |
| `psidot(ρ)` | ∂ψ/∂t (loop voltage) | V |
| `q_face(ρ)` | Safety factor | - |
| `s_face(ρ)` | Magnetic shear | - |
| `j_total(ρ)` | Total current density | A/m² |
| `Ip_profile(ρ)` | Enclosed plasma current | A |
| `sigma(ρ)` | Plasma conductivity | S/m |
| `Z_eff(ρ)` | Effective charge | - |
| `v_loop_lcfs` | Loop voltage at LCFS | V |

Transport coefficients at each step:
- χ_i, χ_e (with ITG/TEM/ETG breakdown)
- D_e, V_e (particle transport)
- Neoclassical contributions

### 3.8 JAX Features

- **JIT compilation**: After first run, subsequent runs are ~10-100x faster
- **Automatic differentiation**: Enables gradient-based optimization and Jacobian computation for Newton-Raphson solver
- **vmap**: Can batch multiple simulations in parallel
- **GPU support**: Can run on GPU for further speedup (requires jax[cuda])
- **Differentiable**: Entire simulation is differentiable end-to-end through JAX

### 3.9 Limitations

- **Fixed geometry per time step**: Geometry is loaded from file, not self-consistently updated (this is what our coupling fixes)
- **1D only**: Radial profiles only — no 2D effects (ballooning, edge localized modes)
- **No MHD stability**: Only sawtooth model (simple q=1 mixing); no tearing modes, kink modes, or other MHD instabilities
- **Pedestal is prescribed**: Uses a simple model (set pedestal height/width), not self-consistent edge physics
- **No SOL/divertor**: Core plasma only
- **JAX learning curve**: Configuration requires Python (not simple YAML), JAX compilation can be slow on first run

---

## 4. Coupling Analysis: FreeGSNKE + TORAX

### 4.1 The Self-Consistent Loop

The coupling follows the standard integrated modeling paradigm used in DINA, JINTRAC, CORSICA, etc.:

```
                    ┌──────────────────────────────────────┐
                    │         Self-Consistent Loop          │
                    │                                      │
  Coil voltages ──► │  FreeGSNKE                           │
  V_PF(t)          │  ┌─────────────────────┐             │
                    │  │ Grad-Shafranov +    │  geometry   │
                    │  │ Circuit equations   │────────────►│
                    │  │ ψ(R,Z), q(ρ),      │             │
                    │  │ boundary, shape     │             │  TORAX
                    │  └──────────▲──────────┘             │  ┌──────────────────┐
                    │             │                         │  │ Transport PDEs    │
                    │             │ p(ρ), j(ρ)             │  │ Te(ρ), ne(ρ),    │
                    │             │ (updated profiles)      │  │ j(ρ), Ti(ρ)     │
                    │             │                         │  └──────────────────┘
                    │             └─────────────────────────│◄──────────┘
                    │                                      │
                    └──────────────────────────────────────┘
```

At each coupling time step:
1. **FreeGSNKE** solves the GS equation + circuit equations → produces ψ(R,Z), q(ρ), plasma boundary, coil currents
2. **Extract geometry** from FreeGSNKE equilibrium → compute flux-surface averages → build TORAX Geometry object
3. **TORAX** evolves transport equations for one (or several) time step(s) using updated geometry → produces Te(ρ), ne(ρ), j(ρ)
4. **Update FreeGSNKE profiles** from TORAX outputs: compute p(ψ) from Te, ne; compute J_tor from j, bootstrap current
5. **Repeat**

### 4.2 The Coupling Interface: GEQDSK

The most practical coupling path uses the **GEQDSK format** as the interchange:

```
FreeGSNKE → write GEQDSK file → TORAX reads GEQDSK file as geometry
TORAX → compute p(ρ), j(ρ) → update FreeGSNKE profile parameters
```

**FreeGSNKE → TORAX direction:**
FreeGS/FreeGSNKE has `geqdsk.write(eq, file)` that produces a standard GEQDSK file.
TORAX has `geometry_type: 'eqdsk'` that reads GEQDSK and constructs `StandardGeometryIntermediates`.

From GEQDSK, TORAX extracts:
- ψ(ρ), Φ(ρ) — poloidal and toroidal flux
- F(ρ) = R·Bt — toroidal field function
- R_in(ρ), R_out(ρ) — flux surface midplane radii
- Flux-surface-averaged quantities: ⟨|∇ψ|²⟩, ⟨B²⟩, ⟨1/R²⟩, etc.
- V'(ρ) — volume derivative
- κ(ρ), δ(ρ) — elongation, triangularity

**TORAX → FreeGSNKE direction:**
From TORAX output profiles, we need to construct FreeGSNKE-compatible pressure and current profiles:
```python
# From TORAX outputs:
Te_rho = torax_output.T_e       # keV, on ρ grid
ne_rho = torax_output.n_e       # m⁻³
Ti_rho = torax_output.T_i       # keV
j_rho = torax_output.j_total    # A/m²

# Compute pressure: p = n_e·T_e + n_i·T_i (in SI)
p_rho = ne_rho * Te_rho * 1.602e-16 + ni_rho * Ti_rho * 1.602e-16  # Pa

# Map from ρ to ψ_N for FreeGSNKE
# Use the ψ(ρ) mapping from the current equilibrium
# Construct pprime(ψ_N) = dp/dψ and ffprime(ψ_N) from j and p

# Feed into FreeGSNKE via ProfilesPprimeFfprime
profiles = ProfilesPprimeFfprime(pprime_func, ffprime_func, fvac)
```

### 4.3 Coupling Strategies

**Option A: Loose coupling (file-based, recommended to start)**
```
for t in time_steps:
    # 1. FreeGSNKE step
    freegsnke_solver.nlstepper(dt, V_coils, profile_params)
    eq = freegsnke_solver.equilibrium

    # 2. Write GEQDSK
    geqdsk.write(eq, f"eq_{t:.3f}.geqdsk")

    # 3. Run TORAX for this interval
    torax_config['geometry']['geometry_file'] = f"eq_{t:.3f}.geqdsk"
    torax_output = run_torax(torax_config, t, t+dt)

    # 4. Extract profiles, update FreeGSNKE profile params
    p, j = extract_profiles(torax_output)
    profile_params = update_profile_params(p, j, eq)
```

Pros: Simple, uses existing I/O. Easy to debug.
Cons: File I/O overhead; coupling time step must be relatively large.

**Option B: In-memory coupling (tighter, better physics)**
```
for t in time_steps:
    # 1. FreeGSNKE step
    freegsnke_solver.nlstepper(dt, V_coils, profile_params)
    eq = freegsnke_solver.equilibrium

    # 2. Extract geometry directly from eq object
    geometry = freegsnke_to_torax_geometry(eq)

    # 3. TORAX step with in-memory geometry
    torax_state = torax_stepper(torax_state, geometry, dt)

    # 4. Update FreeGSNKE profiles from TORAX state
    profile_params = torax_to_freegsnke_profiles(torax_state, eq)
```

Pros: No file I/O; can couple at every time step; tighter physics.
Cons: Need to write `freegsnke_to_torax_geometry()` converter; need to understand TORAX internals for stepping.

### 4.4 The Geometry Converter: FreeGSNKE → TORAX

This is the critical piece of custom code. It must extract flux-surface-averaged quantities from a FreeGSNKE 2D equilibrium and package them into TORAX's `StandardGeometryIntermediates`.

**What we need to compute from ψ(R,Z):**

For each flux surface ψ = const (parameterized by ρ):
1. **Trace the flux surface contour** in (R,Z) space
2. **Compute flux-surface averages** via contour integrals:
   - ⟨|∇ψ|²⟩ = ∮ |∇ψ| dl / ∮ dl/|∇ψ|  (approximately)
   - ⟨1/R²⟩ = ∮ (1/R²) dl/|∇ψ| / ∮ dl/|∇ψ|
   - ⟨B²⟩ = ∮ B² dl/|∇ψ| / ∮ dl/|∇ψ|
3. **Compute V'(ρ)** = dV/dρ from the enclosed volume
4. **Extract R_in, R_out** at Z=Z_axis for each flux surface
5. **Compute elongation, triangularity** from the flux surface shape

**Alternative shortcut:** Write GEQDSK from FreeGSNKE, then use TORAX's existing EQDSK reader. This avoids reimplementing the flux-surface averaging. The EQDSK reader in TORAX (`eqdsk.py`) already handles all the geometric computations.

**Recommendation:** Start with GEQDSK file interchange (Option A), then optimize to in-memory if performance is an issue.

### 4.5 The Profile Converter: TORAX → FreeGSNKE

TORAX outputs profiles on a ρ_tor grid. FreeGSNKE needs p'(ψ_N) and FF'(ψ_N), or equivalently, parametric profile parameters (β_p, Ip, α_m, α_n).

**Approach 1: Fit parametric profiles (simpler)**
```python
# From TORAX: p(ρ), j(ρ)
# Compute global quantities:
beta_p = compute_betap(p_rho, Ip, geometry)
Ip = integrate_j(j_rho, geometry)
# Use ConstrainBetapIp with updated betap, Ip
profiles = ConstrainBetapIp(betap=beta_p, Ip=Ip, fvac=fvac)
```

Pros: Simple, uses existing FreeGSNKE profile classes.
Cons: The Lao parameterization has limited shape flexibility — the actual TORAX profiles may not match the parametric shape.

**Approach 2: Direct p'(ψ) and FF'(ψ) (more accurate)**
```python
# From TORAX: p(ρ), j(ρ), ψ(ρ)
# Compute dp/dψ = (dp/dρ) / (dψ/dρ) numerically
pprime = np.gradient(p_rho, psi_rho)
# Compute FF' from Jtor and p': FF' = μ₀·R·Jtor - R²·p'
ffprime = compute_ffprime(j_rho, pprime, R_rho)
# Use ProfilesPprimeFfprime
profiles = ProfilesPprimeFfprime(
    pprime_func=interp1d(psi_norm, pprime),
    ffprime_func=interp1d(psi_norm, ffprime),
    fvac=fvac
)
```

Pros: Exact representation of TORAX profiles; no information loss.
Cons: Numerical differentiation can be noisy; need smoothing.

**Recommendation:** Start with Approach 1 (parametric fit) for robustness, validate, then switch to Approach 2 if profile shape matters.

### 4.6 Time Scale Considerations

- **Transport time scale** (TORAX): τ_E ~ 1-10 seconds for ITER → dt_transport ~ 0.1-2 s
- **Equilibrium evolution time scale** (FreeGSNKE): τ_eq ~ R·μ₀·σ_wall ~ 0.01-0.1 s (wall time) to ~10 s (current diffusion)
- **Coupling time step**: Should be the minimum of both → dt_couple ~ 0.1-1 s

For the pre-disruption phase (10-100 seconds), this means ~100-1000 coupling steps — feasible in Python.

### 4.7 What We Do NOT Need to Couple

For our ML training data generation, some simplifications are acceptable:
- **No feedback control**: Coil voltages can be prescribed (from ITER scenario files), not feedback-controlled
- **No precise vertical stability**: We only need the plasma to be vertically stable enough to produce a valid trajectory; we don't need ms-accurate VDE modeling
- **No pedestal/ELM dynamics**: A simple prescribed pedestal in TORAX is sufficient
- **Approximate coupling is acceptable**: Small inconsistencies between equilibrium and transport are tolerable for synthetic training data — we need diverse, physics-plausible trajectories, not exact experimental reproductions

---

## 5. Implementation Plan

### Phase 1: Installation & Verification (Days 7-8)

**Task 1.1:** Install FreeGSNKE
```bash
conda create -n tokamak_coupled python=3.10
conda activate tokamak_coupled
pip install "freegsnke[freegs4e]"
pip install torax
```

**Task 1.2:** Run FreeGSNKE static examples
- Run ITER static inverse equilibrium (example8)
- Verify: get ψ(R,Z), q profile, Ip, β_N, separatrix

**Task 1.3:** Run FreeGSNKE evolutive example
- Run MAST-U evolutive example (example5)
- Verify: time-evolving equilibrium, coil currents, plasma current

**Task 1.4:** Run TORAX ITER hybrid scenario
- Run `iterhybrid_rampup.py` config
- Verify: Te(ρ,t), ne(ρ,t), q(ρ,t) evolution over 80s

**Task 1.5:** Verify GEQDSK interchange
- Write GEQDSK from FreeGSNKE equilibrium
- Load in TORAX as geometry
- Compare geometric quantities

### Phase 2: ITER Machine Definition for FreeGSNKE (Days 9-10)

**Task 2.1:** Define ITER machine in FreeGSNKE format
- Active coils: CS (6 modules) + PF1-PF6 (6 coils) = 12 independent circuits
- Passive structures: vacuum vessel, blanket modules (simplified)
- Limiter: first wall contour
- Wall: vacuum vessel inner wall

Sources: ITER design data (public), DINA-IMAS scenario files for coil positions/currents

**Task 2.2:** Compute a reference ITER equilibrium
- Use FreeGSNKE static inverse solver
- Target: 15MA, q95~3, standard ITER shape (κ~1.7, δ~0.33)
- Verify against published ITER equilibrium data

**Task 2.3:** Run an ITER-like time evolution
- Prescribe CS/PF voltages for an Ip ramp from the DINA scenario files
- Evolve from startup to flat-top
- Verify Ip(t), q95(t), β_N(t) trajectories

### Phase 3: FreeGSNKE ↔ TORAX Coupling (Days 11-14)

**Task 3.1:** Implement GEQDSK-based loose coupling
```python
class CoupledSimulator:
    def __init__(self, freegsnke_machine, torax_config):
        self.freegsnke = setup_freegsnke(freegsnke_machine)
        self.torax = setup_torax(torax_config)

    def step(self, dt, V_coils, heating_params):
        # 1. FreeGSNKE step
        self.freegsnke.nlstepper(dt, V_coils, self.profile_params)
        eq = self.freegsnke.equilibrium

        # 2. Write temp GEQDSK
        with tempfile.NamedTemporaryFile(suffix='.geqdsk') as f:
            geqdsk.write(eq, f)
            # 3. TORAX step with new geometry
            self.torax_state = run_torax_step(f.name, self.torax_state, dt)

        # 4. Update profile params
        self.profile_params = self.extract_profile_params()

        return self.get_signals()
```

**Task 3.2:** Implement profile extraction (TORAX → FreeGSNKE)
- Compute β_p from TORAX pressure profile
- Compute Ip from TORAX current density
- Update FreeGSNKE ConstrainBetapIp parameters

**Task 3.3:** Validate coupling
- Run 15MA ITER flat-top for 10 seconds
- Verify Te, ne profiles are physically reasonable
- Verify q profile consistency between FreeGSNKE and TORAX
- Verify energy and particle conservation

### Phase 4: Shot Generation (Days 15-18)

**Task 4.1:** Normal shot generator
- Use ITER scenario waveforms (Ip(t), P_heat(t), n_target(t))
- Run coupled FreeGSNKE+TORAX simulation
- Extract time series: Te(ρ,t), ne(ρ,t), j(ρ,t), q(ρ,t), Ip(t), β_N(t), q95(t), f_GW(t)
- Verify signals stay within safe limits
- Label = 0

**Task 4.2:** Disruptive shot generator
- Start from normal trajectory
- Inject perturbations at random time:
  - **Density limit approach**: Increase ne_edge → f_GW > 0.95
  - **Beta limit approach**: Increase heating → β_N > 3.2
  - **q limit approach**: Reduce Ip or increase density → q95 < 2.2
  - **Impurity injection**: Increase Z_eff → enhanced radiation
- Detect trigger crossing
- Hand off to DREAM for TQ/CQ/RE simulation
- Label = 1

**Task 4.3:** Parameter space sampling
- Vary: Ip (5-15 MA), P_heat (10-40 MW), ne_target (0.5-1.2 × n_GW), Z_eff (1.2-3.0)
- Generate diverse normal + disruption shot pairs
- Target: 1000+ shots total

### Phase 5: HDF5 Dataset & Validation (Days 19-21)

**Task 5.1:** HDF5 writer
```
data/dataset_v1.h5:
  shots/
    shot_00001/
      signals/
        Te          [N_rho × N_t]   float32
        ne          [N_rho × N_t]   float32
        j_tor       [N_rho × N_t]   float32
        q           [N_rho × N_t]   float32
        Ip          [N_t]           float32
        betaN       [N_t]           float32
        q95         [N_t]           float32
        f_GW        [N_t]           float32
      time          [N_t]           float32
      label         scalar          int
      disruption_time  scalar       float  (NaN for normal)
    ...
  metadata/
    generation_config
    code_versions
    parameter_ranges
```

**Task 5.2:** Dataset validation
- Statistical analysis of signal distributions
- Physics consistency checks (energy conservation, q monotonicity)
- Visual inspection of representative shots
- Comparison with published ITER scenario data

---

## References

1. Amorisco et al., "FreeGSNKE: A Python-based dynamic free-boundary toroidal plasma equilibrium solver", Phys. Plasmas 31, 042517 (2024)
2. Pentland et al., "Validation of the static forward Grad-Shafranov equilibrium solvers in FreeGSNKE and Fiesta using EFIT++ reconstructions from MAST-U", Phys. Scr. 100, 025608 (2025)
3. Citrin et al., "TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX", arXiv:2406.06718 (2024)
4. FreeGS documentation: https://freegs.readthedocs.io/
5. FreeGSNKE documentation: https://docs.freegsnke.com/
6. TORAX documentation: https://torax.readthedocs.io/
7. FreeGS GitHub: https://github.com/freegs-plasma/freegs
8. FreeGSNKE GitHub: https://github.com/FusionComputingLab/freegsnke
9. TORAX GitHub: https://github.com/google-deepmind/torax
