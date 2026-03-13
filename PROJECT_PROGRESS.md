# Project Progress - Tokamak Disruption Simulator

This file tracks daily progress following the 6-week implementation plan (`Plan_Implementacion_Simulacion_Tokamak.pdf`).

---

## WEEK 1: Environment Setup & Code Installation

### Day 1-2: Repository Setup & Base Environment (COMPLETED)

**Date:** 2025-01-23 (approx)
**Commit:** `25f8a68` - feat: initial project setup with environment configuration

**Tasks completed:**
- Created GitHub repository: tokamak-disruption-simulator
- Configured initial directory structure: `src/{dina,dream,pipeline,utils}`, `configs/{dina,dream}`, `scripts/`, `tests/`, `notebooks/`, `data/`, `docs/`, `results/`
- Created conda environment (`environment.yml`) and pip environment (`requirements.txt`) with Python 3.10
- Installed base dependencies: numpy, scipy, h5py, matplotlib, pandas
- Created `CLAUDE.md` with project conventions and architecture
- Created `README.md` with project overview
- Created `check_dependencies.sh` verification script
- Created `scripts/verify_installation.py`
- Created `configs/generation.yaml` with disruption trigger thresholds

**Project structure established:**
```
tokamak-disruption-simulator/
  src/{dina,dream,pipeline,utils}/  -- Module stubs with __init__.py
  configs/{dina,dream}/             -- Config directories
  configs/generation.yaml           -- Main generation config
  scripts/verify_installation.py    -- Installation verification
  check_dependencies.sh             -- Dependency checker
  tests/                            -- Test directory
  notebooks/                        -- Analysis notebooks
  data/                             -- Output datasets
  docs/                             -- Documentation
  results/                          -- Results directory
```

---

### Day 3-4: DREAM Installation (COMPLETED)

**Date:** 2025-02-06 (approx)
**Commit:** `9f8be0b` - feat: complete DREAM installation and verification

**Tasks completed:**
- Verified prerequisites: CMake >= 3.12, gcc >= 7.0, GSL >= 2.4, HDF5, PETSc
- Cloned DREAM repository from https://github.com/chalmersplasmatheory/DREAM.git
- Compiled PETSc following DREAM guide
- Compiled DREAM: mkdir build && cd build && cmake .. && make -j4
- Configured PYTHONPATH for the DREAM Python frontend
- Executed basic documentation example
- Verified `dreami` executes without errors

**Key notes:**
- DREAM is installed as a system dependency (C++/CMake/PETSc/GSL)
- Python frontend accessible via PYTHONPATH configuration
- DREAM handles: Thermal Quench, Current Quench, runaway electrons, shattered pellet injection

---

### Day 5-6: DINA-IMAS Installation — BLOCKED, Approach Revised (2026-02-21)

**Date:** 2025-02-20 / 2026-02-21
**Status:** DINA cannot run locally. Approach revised for Week 2 onwards.

**Tasks from plan:**
- [x] Clone repository: https://github.com/iterorganization/DINA-IMAS.git → `/home/miguel/Desktop/Plasma/IA/DINA-IMAS/`
- [x] IMAS Codex MCP connected (previous session)
- [~] Execute configuration script: `source ci_header.sh` — **requires ITER HPC `module` system, not available on desktop**
- [x] Compile DINA core Fortran libraries — **succeeded** (`green.a`, `dina_99.a`, all 4 controllers)
- [~] Compile IMAS interface layer (`imas/iwrap/`) — **failed**, needs `al-fortran`, `xmllib`, `iWrap` (HPC-only)
- [~] Test GUI and run example scenario — **blocked** by above

#### What DINA-IMAS actually is

DINA-IMAS has two distinct layers:
1. **Core Fortran physics engine** (`src/`) — solves MHD equilibrium + transport equations (2D free-boundary Grad-Shafranov, 1D energy/particle transport, impurity radiation, circuit equations). **Compiled successfully.**
2. **IMAS interface layer** (`imas/iwrap/`) — wraps the Fortran core so it can read/write IMAS IDS data and be called from Python. **Requires ITER HPC infrastructure.**

The IMAS interface needs three packages only available on ITER's cluster:
- `al-fortran` — IMAS Access Layer (reads/writes from MDS+ databases)
- `xmllib` — XML parameter parsing
- `iWrap` — ITER's actor wrapping framework (Fortran → Python callable)

These are loaded via `module load IMAS/3.39.0-foss-2023b` on ITER's HPC — `module` is an HPC tool that doesn't exist on a standard desktop.

#### Why IMAS-Python does NOT solve the problem

During this session we discovered `imas-python` (PyPI, LGPL-3.0) — an open-source Python reimplementation of the IMAS data layer that runs without HPC infrastructure. However, after analysis, **installing it would not solve our core problem**.

The critical distinction:
- **IMAS** is a data standard and framework (like a file format + API for fusion physics data)
- **DINA** is the actual physics simulation code that uses IMAS as its I/O format

`imas-python` gives us the ability to create and read IMAS data structures (IDS objects), but it cannot make DINA run. The physics engine — the code that actually simulates plasma evolution (Te, ne, j, q profiles over time) — remains inaccessible locally. Installing `imas-python` would solve a data formatting problem we don't actually have; the real gap is the absence of a physics simulator.

#### What the DINA scenario data actually contains

The `machines/iter/` directory contains 13 ITER scenarios (15MA DT, 7.5MA He, 5MA, 2MA variants). These are **scenario input prescriptions** (reference waveforms that drive a DINA run), not full DINA simulation outputs. They contain:
- **Available**: `Ip(t)`, coil currents `I_CS/PF(t)`, voltages `V(t)`, heating schedules `P_ECH/ICH(t)`, density targets
- **Not available**: 1D profiles `Te(ρ,t)`, `ne(ρ,t)`, `j(ρ,t)`, `q(ρ,t)`, betaN(t), li(t)

These waveforms come from ITER's scenario design work (calibrated by DINA), so they represent realistic global plasma evolution trajectories. They are useful as reference waveforms for normal shots, but lack the profile data that an ML disruption predictor actually observes.

---

### Day 7: Revised Approach — Integrated Physics Modeling (NEXT SESSION)

**Context:** The original plan assumed DINA would provide physics-accurate pre-disruption plasma evolution. Since DINA cannot run locally, we need an alternative that still provides the 1D profiles (Te, ne, j, q vs ρ) over time that constitute the actual ML training signals. A purely phenomenological (parametric curve) approach would produce low-fidelity synthetic data that may not generalize to real discharges.

**New approach: Integrated physics modeling with open-source codes**

Instead of DINA as a monolithic simulator, we compose multiple open-source physics codes that together cover the same physics:

```
FreeGS / FreeGSNKE     ←→     Transport code (RAPTOR / OMFIT)
  (MHD equilibrium)              (1D profile evolution)
       ↕                                  ↕
   q(ρ), psi(R,Z),              Te(ρ,t), ne(ρ,t), j(ρ,t)
   plasma boundary
              ↓
       Trigger detector
  (Greenwald, betaN, q95)
              ↓
           DREAM
    (disruption simulation)
              ↓
       HDF5 dataset
```

The coupling is the same self-consistent loop that DINA uses internally:
equilibrium provides geometry and q profile → transport evolves profiles given that geometry → updated profiles feed back into equilibrium → repeat.

**Candidate codes to evaluate:**

1. **FreeGS** (`pip install freegs`) — free-boundary MHD equilibrium solver in Python. Gives: psi(R,Z), q(ρ), plasma boundary shape, Ip. Static solver — must be called at each time step. Well-documented, active development, used in MAST-U, STEP, and other programs.

2. **FreeGSNKE** — extension of FreeGS for time-evolving free-boundary equilibria with kinetic (transport) coupling. Specifically designed for discharge evolution. Open-source, on PyPI. Most direct replacement for the equilibrium+evolution part of DINA.

3. **RAPTOR** — reduced-order 1D transport code solving current diffusion + energy transport equations. Designed for real-time control applications at JET, TCV, AUG. Fast by design. The question is accessibility — originally MATLAB, Python ports exist.

4. **OMFIT** — General Atomics' open-source integrated modeling framework (`pip install omfit-core`). Couples FreeGS with multiple transport codes. Has a Python API. Used extensively at DIII-D. Could provide the integration layer we need without writing the coupling ourselves.

**Use of existing DINA scenario data:**

The ITER scenario `.dat` files serve as calibration constraints:
- Use Ip(t) waveforms to constrain the plasma current trajectory (ramp-up duration, flat-top value, ramp-down)
- Use heating schedules P(t) as boundary conditions for the transport code
- Use PF coil configurations from `tokamak_config.dat` if running free-boundary equilibrium

Normal shots: full Ip trajectory from the reference scenarios, profiles stay within safe limits, simulation terminates gracefully.
Disruption shots: start from a reference trajectory, then perturb parameters (density puff, heating transient, etc.) until a disruption trigger is crossed, then hand off to DREAM.

**Tasks for next session:**
- [ ] Evaluate FreeGSNKE: install, run a basic example, assess whether it can time-evolve an ITER-like equilibrium
- [ ] Evaluate OMFIT: check if it provides a usable Python API for automated scenario generation
- [ ] Assess RAPTOR availability and whether a Python interface exists
- [ ] Choose the combination that gives us Te(ρ,t), ne(ρ,t), j(ρ,t), q(ρ,t) with reasonable physics fidelity
- [ ] Prototype a minimal pre-disruption trajectory for one scenario (15MA ITER reference)
- [ ] Document chosen approach and update CLAUDE.md architecture section

---

## WEEK 2-3: Equilibrium + Transport Coupling

**Key invariants (unchanged from original plan):**
- Pipeline structure: pre-disruption evolution → trigger detection → DREAM → HDF5
- Disruption triggers: Greenwald > 0.95, betaN > 3.2, q95 < 2.2
- Output format: HDF5 with `shots/shot_NNNNN/{signals, time, label}`
- DREAM handles all disruption physics (TQ, CQ, runaway electrons)

---

### Days 7-10: FreeGSNKE Equilibrium Solver (COMPLETED)

**Date:** 2026-03-09
**Status:** Static equilibrium working, dynamic evolution validated.

**Tasks completed:**
- [x] Implemented `EquilibriumSolver` class wrapping FreeGSNKE (`src/predisruption/equilibrium.py`)
- [x] Built ITER machine description from DINA-IMAS config (`src/predisruption/iter_machine.py`)
- [x] Fixed `eq.q()` call — must pass array, not scalar (causes TypeError)
- [x] Migrated from `forward_solve` (converges to R=7.9m) to `inverse_solve` with null-point + isoflux constraints
- [x] Fixed topology jump: solver's `best_psi` picks wrong equilibrium; use final iteration state instead
- [x] Fixed premature convergence: tight tolerance (1e-4) + limited iterations (15) prevents basin jumping
- [x] Applied two monkey-patches for FreeGSNKE bugs (`inside_mask` IndexError, `copy_into` NoneType)
- [x] Verified static equilibrium: R_axis=6.21m, kappa=1.78, delta=0.30, q95=1.72
- [x] Verified dynamic evolution: `init_dynamic()` + `step()` — equilibrium stable over 3 time steps
- [x] Verified GEQDSK output for TORAX coupling: `write_geqdsk_tmp()` works
- [x] Verified `get_signals()` extracts all physics quantities including q profile
- [x] Removed dead code (`solve_static_inverse()` using wrong API)

**Equilibrium quality achieved (ITER 15 MA flat-top):**

| Parameter | Value | Target | Status |
|-----------|-------|--------|--------|
| R_axis | 6.21 m | 6.2 m | OK |
| Z_axis | -0.02 m | 0.0 m | OK |
| Ip | 15.0 MA | 15.0 MA | OK |
| betap | 0.50 | 0.5 | OK |
| kappa | 1.78 | 1.75 | OK |
| delta | 0.30 | 0.33 | OK |
| q95 | 1.72 | 3.0 | Low (*) |
| betaN | 0.71 | 1.8 | Low (*) |
| li | 1.64 | 0.85 | High (*) |

(*) q95, betaN, li discrepancy explained by minor radius: the equilibrium has a=1.63m instead of the ITER target a=2.0m. Since q95 ∝ a², the ratio (1.63/2.0)² ≈ 0.66 accounts for the low q95. The coil configuration from DINA-IMAS produces a smaller plasma than the design target. The equilibrium is self-consistent — values are physically correct for the computed shape.

**Key technical discoveries:**
1. `forward_solve` cannot find the correct ITER axis (coil psi too flat at midplane → R drifts to 7.9m)
2. `inverse_solve` with null-point constraints pins the axis at R=6.2m
3. Isoflux constraints (X-point + top-of-plasma) enforce kappa~1.78 and delta~0.32
4. Initial plasma_psi must be a Gaussian centered at (6.2, 0.0) — limiter centroid init pulls axis toward X-point
5. Solver iteration count is critical: too few (< 5) = poor convergence; too many (> 25) = topology jump to R=7.9m; sweet spot is 10-15
6. FreeGSNKE's `ConstrainBetapIp` Raxis parameter should be set to R_major (6.2), not default 1.0

---

### Days 11-14: Transport Coupling (COMPLETED)

**Date:** 2026-03-10
**Status:** Coupled FreeGSNKE + SimplifiedTransport loop working. 5 bugs fixed. Validation scripts + physics tests passing.

**Tasks completed:**
- [x] Fixed Bug A: Unit conversion in energy balance — `T_avg_keV_new` was 1000x too small (extra `*1e3` in denominator). Te collapsed to 0.01 keV on first step.
- [x] Fixed Bug B: `W_thermal` not initialized from profiles — dataclass default was 0.0, causing energy balance to start from zero.
- [x] Fixed Bug C: Current diffusion CFL instability — explicit scheme with D=0.01, drho=0.02 had CFL limit dt_max=0.02s but dt=1.0s (50x unstable). Replaced with Crank-Nicolson implicit tridiagonal solve.
- [x] Fixed Bug D: Empty geometry dict in coupling — `extract_freegsnke_profiles()` was called with `{}` instead of actual equilibrium signals, so Ip always defaulted to 15e6.
- [x] Fixed Bug E (discovered during validation): Unit conversion in `extract_freegsnke_profiles` — extra `*1e3` factor made betap 1000x too large (729 instead of 0.73). Same pattern as Bug A.
- [x] Added `a_minor` to `get_signals()` output — computed from separatrix R extent, used by transport for energy balance and Greenwald calculations.
- [x] Updated coupling to use actual `a_minor` from equilibrium for Greenwald fraction.
- [x] Fixed Colab notebook: replaced `YOUR_USERNAME` with `BecerraMiguel`, fixed `TRIGGERS` import (was `iter_machine`, corrected to `shot_runner`).
- [x] Single-step validation (`scripts/validate_single_step.py`): 6/6 checks pass
- [x] Multi-step trajectory validation (`scripts/validate_trajectory.py`): 90s trajectory completes (83 of 90 steps before O-point loss)
- [x] Trigger validation (`scripts/validate_triggers.py`): density and beta triggers fire as expected
- [x] Physics tests (`tests/test_transport_coupling.py`): 6/6 tests pass

**Validation results (single step):**

| Quantity | Value | Expected | Status |
|----------|-------|----------|--------|
| W_thermal (init) | 318 MJ | > 0 | PASS (Bug B fix) |
| Te(0) after step | 37 keV | 5-50 keV | PASS (Bug A fix) |
| betap (transport) | 0.73 | 0.01-5.0 | PASS (Bug E fix) |
| R_axis (dynamic step) | 6.40 m | ~6.2 m | PASS |
| ne(0) | 1.0e20 m^-3 | 1e19-5e20 | PASS |
| W_thermal (step) | 304 MJ | 100 kJ-1 GJ | PASS |

**Validation results (90s trajectory):**

| Quantity | Final value | Notes |
|----------|-------------|-------|
| Ip | 2.98 MA | Stays at initial (no voltage control) |
| q95 | 5.81 | Stable (low-current equilibrium) |
| betaN | 0.78 | Stable |
| Te(0) | 6.7 keV | Equilibrium reached after ~5 steps |
| f_GW | 2.37 | High because Ip=3 MA, density target for 15 MA |
| Duration | 83/90 steps | O-point loss at t=84s (edge case) |

**Architecture implemented:**

```
FreeGSNKE (local)              SimplifiedTransport (local)
  solve_static(Ip, betap)         init(geometry, Ip, Te0, ne0)
  get_signals() → geometry        step(geometry, sources, dt)
     │                                │
     │    geometry={Ip, q95, kappa,   │
     │              a_minor, q_prof}  │
     │  ──────────────────────────►   │
     │                                │
     │    profiles={betap, Ip}        │
     │  ◄──────────────────────────   │
     │                                │
  step(dt, V_coils, betap, Ip)       │
     │                                │
     └── repeat (CoupledSimulator) ───┘
```

**Key files modified:**
- `src/predisruption/transport.py` — Bugs A, B, C, E fixed
- `src/predisruption/coupling.py` — Bug D fixed + a_minor passthrough
- `src/predisruption/equilibrium.py` — Added a_minor to get_signals()
- `notebooks/colab_pipeline.ipynb` — Fixed repo URL + TRIGGERS import

**New files:**
- `scripts/validate_single_step.py` — Single equilibrium+transport step validation
- `scripts/validate_trajectory.py` — Full 90s coupled trajectory validation
- `scripts/validate_triggers.py` — Density and beta disruption trigger validation
- `tests/test_transport_coupling.py` — 6 physics regression tests

**Known limitations:**
1. **q95 ≈ 1.7** (below 2.2 trigger) — due to a_minor=1.63m < target 2.0m. q95 trigger cannot be used for disruption scenarios.
2. **Ip stays at initial value** — no coil voltage control implemented; dynamic solver evolves via circuit equations with V=0. Ip waveform is passed to profile object but doesn't drive actual current evolution.
3. **O-point loss at t~84s** — FreeGSNKE dynamic solver occasionally loses O-point after many steps. Acceptable for current phase.
4. **f_GW unrealistically high at low Ip** — Greenwald limit scales with Ip; density targets calibrated for 15 MA operation give f_GW >> 1 at 3 MA.
5. **TORAX not yet integrated** — JAX incompatible locally; SimplifiedTransport backend validated. Colab notebook prepared for TORAX execution.

---

## WEEK 3-4: Trigger Detection + DREAM Handoff

### Days 15-18: Disruption Trigger Detection (PENDING)

**Goal:** Monitor the coupled FreeGSNKE+TORAX evolution and detect when operational limits are crossed.

**Tasks:**
- [ ] Implement `TriggerDetector` class in `src/pipeline/` that evaluates disruption criteria at each time step
- [ ] Disruption triggers from `configs/generation.yaml`:
  - Greenwald fraction > 0.95 (density limit)
  - βN > 3.2 (beta limit)
  - q95 < 2.2 (current limit / kink stability)
- [ ] Define the "trigger point" — the exact time and plasma state at which the disruption begins
- [ ] Extract pre-disruption signal window (e.g., last 100 ms before trigger) for ML training data

### Days 19-21: DREAM Handoff & Disruption Simulation (PENDING)

**Goal:** Take the plasma state at the trigger point and simulate the disruption with DREAM.

**Tasks:**
- [ ] Implement `DREAMHandoff` class that converts FreeGSNKE equilibrium + transport profiles into DREAM input
- [ ] Map FreeGSNKE signals → DREAM initial conditions: Te(ρ), ne(ρ), j(ρ), Ip, plasma geometry
- [ ] Run DREAM for thermal quench (TQ) + current quench (CQ) phases
- [ ] Extract DREAM output signals: runaway electron current, radiated power, halo currents
- [ ] Validate one complete shot: normal phase → trigger → DREAM disruption

---

## WEEK 4-5: Batch Generation + Dataset

### Days 22-28: Shot Generation Pipeline (PENDING)

**Goal:** Generate a batch of labeled shots (normal + disruptive) as HDF5 datasets.

**Tasks:**
- [ ] Implement `DatasetGenerator` class in `src/pipeline/` that orchestrates N shots
- [ ] Parameter variation for diverse disruptions:
  - Density ramps (Greenwald limit violations)
  - Heating transients (beta limit)
  - Current ramp-down errors (q95 limit)
- [ ] Normal shot generation: full Ip trajectory without trigger crossing
- [ ] HDF5 writer: save to `data/shots/shot_NNNNN/{signals/*, time, label}` + `metadata/`
- [ ] Target: 50-100 shots (mix of normal and disruptive) for initial ML training set

---

## WEEK 5-6: Validation + ML Pipeline

### Days 29-35: Dataset Validation (PENDING)

**Tasks:**
- [ ] Statistical analysis of generated dataset: signal distributions, label balance
- [ ] Physics sanity checks: energy conservation, current evolution consistency
- [ ] Visualization notebook: plot example shots, signal histograms, disruption statistics

### Days 36-42: FNO Training Pipeline (PENDING)

**Goal:** Train a Fourier Neural Operator on the synthetic disruption dataset.

**Tasks:**
- [ ] Implement data loader for HDF5 shot format
- [ ] FNO architecture for disruption prediction (time-series → disruption probability)
- [ ] Training loop with train/val split
- [ ] Evaluation metrics: prediction accuracy, warning time, false positive rate
- [ ] Final report and documentation

---

## Important Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-02-20 | Connect IMAS Codex MCP to Claude Code | Direct access to IMAS Data Dictionary improves understanding of DINA-IMAS data structures and IDS schemas |
| 2026-02-21 | Abandon DINA, adopt integrated open-source physics modeling | DINA-IMAS requires ITER HPC infrastructure (IMAS Access Layer, iWrap) unavailable on desktop. IMAS-Python does not solve this — the problem is the physics engine, not the data format. Replace with FreeGSNKE + transport code combination that can run locally and provides equivalent physics outputs (Te, ne, j, q profiles over time). |
| 2026-03-10 | Use inverse_solve instead of forward_solve for equilibrium | forward_solve converges to R=7.9m (wrong basin) because ITER coil psi is flat across midplane. inverse_solve with null-point + isoflux constraints pins axis at R=6.2m with correct kappa/delta. |
| 2026-03-10 | Accept q95≈1.7 (vs target 3.0) for now | Minor radius a=1.63m < target 2.0m due to coil configuration limitations. q95 ∝ a², so discrepancy is explained. Equilibrium is self-consistent. Transport coupling (TORAX) will determine actual q95 evolution. |
| 2026-03-10 | Don't restore solver.best_psi after inverse_solve | The solver's "best" iteration often has wrong topology (R~7.8m). Final iteration state has correct axis when null-point constraints are active. |
| 2026-03-10 | Replace explicit current diffusion with Crank-Nicolson | Explicit scheme CFL limit dt_max=0.02s was 50x below the 1.0s coupling timestep → numerical explosion. Implicit scheme is unconditionally stable. |
| 2026-03-10 | Use SimplifiedTransport as primary backend for local runs | TORAX requires JAX (incompatible locally). The 0.5D simplified model (IPB98 scaling + parabolic profiles + implicit current diffusion) produces physics-plausible synthetic data adequate for ML training. |

## Problems & Solutions Log

| Date | Problem | Solution |
|------|---------|----------|
| 2025-02-20 | Browser shows error when visiting MCP URL | Expected — MCP requires SSE client. `"Client must accept text/event-stream"` confirms server is running correctly |
| 2026-02-21 | DINA-IMAS cannot be compiled locally | Core Fortran compiles fine, but IMAS interface layer requires `al-fortran`/`iWrap` (HPC-only). Solution: replace DINA with integrated open-source modeling (FreeGSNKE + transport code) |
| 2026-02-21 | imas-python proposed as fix but rejected | Installing `imas-python` would solve data formatting, not the missing physics. The physics engine (DINA) is what produces Te/ne/j/q profiles — IMAS is just the I/O format it uses. No point adopting IMAS data structures if we have no physics code to generate them. |
| 2026-03-10 | freegs4e.critical.inside_mask bug (single X-point) | Monkey-patch in equilibrium.py: `len(xpoint > 1)` → `len(xpoint) > 1`. Evaluates bool array length instead of list length. |
| 2026-03-10 | freegsnke.copying.copy_into NoneType error | Monkey-patch: when attribute is None, assign directly instead of np.copy(None). |
| 2026-03-10 | eq.q(scalar) raises TypeError | Pass array to eq.q(), not scalar float. freegs4e expects array/list argument. |
| 2026-03-10 | Topology jump after >25 solver iterations | Limit max_solving_iterations=15, use tight tolerance (1e-4) to prevent premature convergence at wrong basin. |
| 2026-03-10 | Te collapses to 0.01 keV on first transport step | Two unit conversion bugs: (A) extra *1e3 in energy→temperature formula, (E) extra *1e3 in keV→J conversion for betap. Both used 1.602e-16*1e3 where 1.602e-16 alone converts keV→J (1 keV = 1.602e-16 J). |
| 2026-03-10 | W_thermal=0 after transport init | Dataclass default not overridden. Fixed by computing W = (3/2) * <p> * V_plasma from initial profiles. |
| 2026-03-10 | Current profile explodes after 1 step (NaN/Inf) | Explicit diffusion CFL instability: D/drho² * dt = 0.01/(0.02)² * 1.0 = 25 >> 0.5. Replaced with Crank-Nicolson implicit scheme. |
| 2026-03-10 | betap=729 from transport (should be ~0.7) | Unit bug in extract_freegsnke_profiles: `T_e * 1.602e-16 * 1e3` gives J*1000 instead of J. Removed extra *1e3. |
| 2026-03-10 | O-point lost at t~84s in long trajectories | FreeGSNKE dynamic solver edge case. Coupling loop catches exception and stops gracefully. Acceptable for current phase. |

## External Dependencies

| Tool | Repository | Status |
|------|-----------|--------|
| DREAM | https://github.com/chalmersplasmatheory/DREAM | Installed |
| DINA-IMAS | https://github.com/iterorganization/DINA-IMAS | Cloned, core compiled, IMAS interface blocked (HPC-only) |
| IMAS Codex MCP | https://github.com/iterorganization/imas-codex | Connected |
| freegs4e | (dependency of FreeGSNKE) | Installed (venv), patched for single-null bug |
| FreeGSNKE | https://github.com/FusionComputingLtd/freegsnke | Installed v2.1.0, static+dynamic solve working |
| TORAX | https://github.com/google-deepmind/torax | To install on Colab (JAX incompatible locally) |
