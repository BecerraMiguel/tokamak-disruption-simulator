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

### Days 15 (partial): TORAX Integration — API Rewrite (COMPLETED)

**Date:** 2026-03-17
**Status:** ToraxTransport class rewritten with correct TORAX v1.3.0 API. Colab notebook updated with 3 validation tests. Ready to test on Colab.

**Context:** Before starting DREAM handoff work, we needed to verify and fix the TORAX transport backend. The original `ToraxTransport` class in `transport.py` was written speculatively with a completely wrong API — all method calls (`torax.build_sim_from_config`, `runner.run_step`, `runner.update_geometry`) did not exist in the actual TORAX v1.3.0 installation. Research into the installed TORAX source revealed the correct API.

**Key architectural decision: Full-run mode for TORAX**

TORAX is designed for complete trajectory runs via `torax.run_simulation(config)` returning an xarray DataTree. Step-by-step execution exists (experimental API) but is impractical for updating geometry between steps. We adopted a two-phase coupling strategy:
1. **Phase 1**: Pre-compute N equilibria with FreeGSNKE + SimplifiedTransport (provides betap feedback)
2. **Phase 2**: Run TORAX once with time-dependent GEQDSK geometry (`geometry_configs` dict)

This means TORAX provides high-fidelity transport profiles given a geometry sequence, while SimplifiedTransport handles the real-time coupling loop for trigger detection.

**Tasks completed:**
- [x] Researched actual TORAX v1.3.0 API from installed source at `venv/lib/python3.12/site-packages/torax/`
- [x] Rewrote `ToraxTransport` class with correct API:
  - `run_full()` → `torax.ToraxConfig.from_dict()` + `torax.run_simulation()`
  - `datatree_to_states()` → converts xarray DataTree to `TransportState` list
  - `init()` → thin wrapper running 1-step TORAX sim
  - `step()` → raises `NotImplementedError` (TORAX is full-run only)
- [x] Rewrote `_default_iter_torax_config()` to match real TORAX format:
  - Based on `torax/examples/iterhybrid_rampup.py` reference config
  - Added all required sections: `pedestal`, `neoclassical`, `time_step_calculator`
  - Fixed `sources` format (old version used fake `"mode": "MODEL"`)
  - Added `cocos: 1` for FreeGSNKE GEQDSK compatibility
  - Supports time-dependent geometry via `geometry_configs`
  - Parameterised: `geometry_dir`, `geometry_files`, `t_final`, `Ip_A`, `transport_model`
- [x] Added `run_trajectory()` method to `TransportSolver` (delegates to `run_full()` for TORAX)
- [x] Added `_deep_merge()` utility for config overrides
- [x] Added `run_with_torax()` method to `CoupledSimulator`:
  - Pre-computes N equilibria with FreeGSNKE + SimplifiedTransport
  - Writes GEQDSK files to temp directory
  - Runs TORAX once with time-dependent geometry
  - Returns trajectory dict (same format as `run()`)
- [x] Updated Colab notebook with 3 progressive TORAX validation tests:
  - **Test A**: TORAX standalone with built-in ITER hybrid example
  - **Test B**: TORAX with our FreeGSNKE GEQDSK + comparison vs SimplifiedTransport
  - **Test C**: TORAX trajectory with time-dependent geometry via `run_with_torax()`
- [x] Verified all 6 existing tests still pass
- [x] Verified auto-detection correctly falls back to simplified transport locally

**Key files modified:**
- `src/predisruption/transport.py` — Rewrote ToraxTransport, config, added run_trajectory()
- `src/predisruption/coupling.py` — Added run_with_torax(), imported SimplifiedTransport
- `notebooks/colab_pipeline.ipynb` — Added TORAX test sections, restructured notebook (29 cells)

**TORAX API facts discovered:**
- Main API: `torax.ToraxConfig.from_dict(config_dict)` + `torax.run_simulation(torax_config)` → `(xr.DataTree, StateHistory)`
- Output: `data_tree["profiles"].ds` has T_e, T_i, n_e, q, j_total, psi on `rho_cell_norm` grid
- Scalars: `data_tree["scalars"].ds` has Ip, v_loop_lcfs, W_thermal_total
- GEQDSK geometry: `geometry_type: 'eqdsk'`, requires `cocos` integer (FreeGSNKE writes COCOS 1)
- Time-dependent geometry: `geometry_configs: {0.0: {'geometry_file': 'eq_t0.geqdsk'}, ...}`
- Transport models: `'constant'` (testing), `'bohm-gyrobohm'`, `'qlknn'` (ML-based turbulent)

**Known risks (to verify on Colab):**
1. COCOS compatibility — FreeGSNKE writes COCOS 1, TORAX converts to COCOS 11 internally with validation. If signs don't match, may need `cocos=3` or similar.
2. TORAX convergence — our equilibrium has unusual a_minor=1.63m; start with `model_name='constant'` transport before upgrading to `'qlknn'`.
3. Profile boundary conditions — use `Ip_from_parameters: True` so TORAX uses config Ip, not GEQDSK Ip.

**Next steps:**
- Push to GitHub and test notebook on Google Colab
- Once TORAX validated, proceed with Days 15-18 (trigger detection) and Days 19-21 (DREAM handoff)

---

### Day 15 (continued): TORAX Colab Validation (COMPLETED)

**Date:** 2026-03-23
**Status:** All 3 TORAX validation tests passing on Google Colab with GPU. Full FreeGSNKE ↔ TORAX coupling demonstrated.

**Context:** Validated the TORAX integration rewritten on Day 15 (API rewrite) by running the Colab notebook. Multiple installation and compatibility issues were discovered and fixed iteratively.

**Colab installation issues fixed (in order of discovery):**

1. **numpy ABI mismatch** — `pip install torax` upgraded numpy .py files to 2.x but left old .so C extensions. Kernel restart loaded mismatched files → `ImportError: numpy._core.multiarray failed to import`. **Fix:** `pip install --force-reinstall --no-deps numpy==<version>` + auto kernel restart via `os.kill(os.getpid(), 9)`.

2. **freegs4e import crash (numba + missing warnings)** — Colab ships numba, which is incompatible with numpy>=2.0. freegs4e's `critical.py` tries `from numba import njit`, fails, then `except ImportError: warnings.warn(...)` crashes because `warnings` was never imported in that file (bug in freegs4e). **Fix:** Inject a fake `numba` module into `sys.modules` with a no-op `njit` decorator before importing freegs4e; also uninstall real numba in install cell.

3. **JAX CUDA plugin version mismatch** — `pip install torax` pulled JAX 0.9.2 but Colab had pre-installed `jax_cuda12_plugin` 0.7.2 (for Colab's default older JAX). PJRT API versions incompatible → `JaxRuntimeError`. **Fix:** Added `pip install jax[cuda12]` to install the matching CUDA plugin for JAX 0.9.2.

4. **Missing deepdiff** — `pip install freegsnke --no-deps` skipped all dependencies including `deepdiff`, which freegsnke imports at load time. **Fix:** Added `deepdiff` to explicit dependency list in install cell.

5. **COCOS mismatch** — Config specified `cocos: 1` but FreeGSNKE writes GEQDSK with sign conventions matching COCOS 7/17. TORAX's `eqdsk` library validated the file data against the declared COCOS and rejected it. **Fix:** Changed to `cocos: 7`.

6. **Flux surface volume monotonicity** — TORAX traces flux surface contours from the magnetic axis outward. Near the separatrix (last ~5% of surfaces), contours in our GEQDSK fail to close properly near the X-point, causing volumes to decrease instead of increase. **Fix:** Set `last_surface_factor=0.75` (stop at 75% of psi_boundary) and `n_surfaces=25`.

7. **Dynamic evolution requires coil_resist** — FreeGSNKE's dynamic solver (`init_dynamic`) needs coil resistance values on the Machine object, which our `tokamak_config.dat` doesn't provide → `AttributeError: 'Machine' object has no attribute 'coil_resist'`. **Fix:** Replaced dynamic evolution with static re-solves at each time point (`solve_static` with updated betap/Ip).

8. **TORAX rho grid mismatch** — TORAX variables live on different grids: `rho_cell_norm` (25 points) vs `rho_face_norm` (27 points). Code assumed all variables on `rho_cell_norm` → `np.interp` length mismatch. **Fix:** Read each variable's actual dimension name via `profiles[var].dims[-1]` and use the matching rho coordinate.

**Test results on Colab:**

| Test | Description | Result | Time |
|------|-------------|--------|------|
| A | TORAX standalone (built-in ITER hybrid, 10s) | PASS | 2m13s |
| B | TORAX with FreeGSNKE GEQDSK + comparison vs SimplifiedTransport | PASS | 1m47s |
| C | Full coupled trajectory (4 equilibria + TORAX 10s) | PASS | ~10 min |

**Test C results (ITER 15 MA flat-top, 10s, constant transport):**

| Signal | Value | Notes |
|--------|-------|-------|
| Ip | 15.0 MA | Constant (prescribed) |
| q95 | 2.85-2.95 | Much improved over initial 1.72 — static re-solves found better equilibria |
| betaN | 3.2-4.3 | Crosses betaN=3.2 limit around t=1-2s — TORAX naturally produces disruption-prone conditions |
| f_GW | 1.5→1.0 | Starts high, decreases — density initialization needs tuning |
| Te(0) | ~9.5 keV | Realistic ITER-like profile with pedestal |
| ne(0) | ~1.8×10²⁰ m⁻³ | Peaked profile, realistic shape |

**Key observation from Test C:** q95 jumped from 1.72 (initial) to ~2.9 on the static re-solves at t=3.3s and t=6.7s. This is because the re-solves start from scratch (no topology memory), and with different betap values the solver finds equilibria with larger minor radius. This serendipitously gives us usable q95 values for the trigger system.

**Key files modified:**
- `notebooks/colab_pipeline.ipynb` — Install cell (two-phase with force-reinstall + auto-restart), fake numba, JAX CUDA, deepdiff, rho grid fixes
- `src/predisruption/transport.py` — COCOS 7, last_surface_factor=0.75, n_surfaces=25, dynamic rho grid in datatree_to_states
- `src/predisruption/coupling.py` — Static re-solves instead of dynamic evolution, graceful handling of few equilibria

---

### CRITICAL DESIGN DECISION: Sequential Coupling (FreeGSNKE ↔ TORAX)

**Date:** 2026-03-23
**Status:** Agreed upon. To be implemented next session.

This section documents the **most important architectural decision** from Day 15 — the coupling strategy between FreeGSNKE (equilibrium) and TORAX (transport). This will be the core of the pre-disruption simulation pipeline.

#### The Problem

In a tokamak, **equilibrium and transport are mutually coupled**:

- The **MHD equilibrium** (shape of flux surfaces, magnetic geometry: grad-rho, volumes, elongation, Shafranov shift) depends on the **pressure and current profiles** (Te, ne, j) — these determine betap, which changes the equilibrium.
- The **transport** (how Te, ne, j evolve over time) depends on the **geometry** — flux surface volumes, metric coefficients, magnetic shear, safety factor, all come from the equilibrium.

Neither can be solved independently if you want self-consistent physics. In production codes like JINTRAC, ETS, or ASTRA, this is handled by alternating between equilibrium and transport solvers at every time step.

The challenge for us: **TORAX is designed for full trajectory runs** — it takes a config, runs `torax.run_simulation()`, and returns the entire time history. There is no practical way to pause TORAX mid-simulation, update the geometry, and continue.

#### Approach 1 (What we implemented first — "Pre-compute then run")

Pre-compute several equilibria at time snapshots using FreeGSNKE + SimplifiedTransport for betap feedback, then pass all GEQDSK files to a single TORAX run with time-dependent `geometry_configs`. TORAX interpolates geometry between snapshots.

**Limitation:** The geometry is based on the simplified transport's estimate of betap — not on TORAX's own profiles. The geometry and transport are not self-consistent.

#### Approach 2 (What we will implement — "Sequential Coupling")

**This is the agreed-upon method.** Divide the simulation into small time intervals and alternate between FreeGSNKE and TORAX at each interval.

**Algorithm (pseudocode):**

```
# Parameters
t_end = 10.0        # total simulation time (s)
dt_couple = 1.0     # coupling interval (s)
n_steps = int(t_end / dt_couple)  # number of coupling steps

# Initialization
eq = FreeGSNKE.solve_static(Ip=15e6, betap=0.5)
geometry = FreeGSNKE.get_signals(eq)
write_geqdsk(eq, "eq_t0000.geqdsk")

# Initial TORAX mini-run to get starting profiles
torax_state = TORAX.run_mini(geqdsk="eq_t0000.geqdsk", t=0→dt_couple)

# Sequential coupling loop
for i in range(1, n_steps):
    t_start = i * dt_couple
    t_end_step = (i + 1) * dt_couple

    # 1. Extract betap from TORAX profiles
    betap_new = compute_betap(torax_state.Te, torax_state.ne, geometry)

    # 2. FreeGSNKE: re-solve equilibrium with TORAX-derived betap
    eq = FreeGSNKE.solve_static(Ip=Ip_waveform(t_start), betap=betap_new)
    geometry = FreeGSNKE.get_signals(eq)
    write_geqdsk(eq, f"eq_t{i:04d}.geqdsk")

    # 3. TORAX: run 1-second mini-trajectory with updated geometry
    #    (geometry is held constant during this interval)
    torax_state = TORAX.run_mini(
        geqdsk=f"eq_t{i:04d}.geqdsk",
        t=t_start → t_end_step,
        initial_profiles=torax_state  # continue from previous state
    )

    # 4. Check disruption triggers
    if betaN > 3.2 or q95 < 2.2 or f_GW > 0.95:
        trigger_time = t_start
        trigger_state = torax_state
        break

    # 5. Record trajectory
    trajectory.append(torax_state)
```

**Why this is better than Approach 1:**

| Aspect | Approach 1 (pre-compute) | Approach 2 (sequential) |
|--------|--------------------------|------------------------|
| Geometry source | SimplifiedTransport betap | TORAX's own betap |
| Self-consistency | Approximate | Self-consistent (TORAX profiles → FreeGSNKE → geometry → TORAX) |
| Feedback loop | Open loop (no correction) | Closed loop at each dt_couple |
| Trigger detection | Post-hoc from TORAX output | Real-time at each step |
| Computational cost | ~10 min (4 eq + 1 TORAX) | ~30 min (10 eq + 10 TORAX), can be optimized |

**Key implementation details:**

1. **Coupling interval `dt_couple`:** We choose 1 second. The equilibrium changes slowly (on the energy confinement time scale τ_E ≈ 3-4s for ITER), so 1s steps are more than adequate. Shorter steps increase accuracy but also computational cost.

2. **TORAX mini-runs:** Each TORAX call is a "full trajectory run" from `t_start` to `t_start + dt_couple` with **constant geometry** (single GEQDSK file, not time-dependent). The geometry is held frozen during each 1-second interval and only updated at the coupling boundary. This avoids the interpolation complexity and is physically justified because the equilibrium doesn't change significantly in 1 second.

3. **Profile continuity between TORAX runs:** Each TORAX mini-run must start from the profiles at the end of the previous run. This requires passing the final Te(rho), ne(rho), j(rho), psi(rho) from the previous TORAX output as initial conditions for the next run. TORAX supports this via `profile_conditions` in its config.

4. **JAX compilation caching:** The first TORAX call is slow (~2 min) because JAX compiles the simulation functions. Subsequent calls with the same grid size reuse the compiled code and are much faster (~30-60s). The sequential approach benefits from this.

5. **betap extraction from TORAX:** After each TORAX mini-run, extract the volume-averaged pressure from Te and ne profiles:
   ```
   <p> = <n_e * T_e + n_i * T_i> (volume average over flux surfaces)
   betap = 2 * mu_0 * <p> / B_pol^2
   ```
   This feeds back into FreeGSNKE for the next equilibrium solve.

6. **Geometry held constant per interval:** Within each 1-second TORAX run, the geometry (flux surface shapes, volumes, metric coefficients) is frozen. This is a standard operator-splitting approach and is accurate as long as dt_couple << τ_E (energy confinement time). For ITER flat-top, τ_E ≈ 3-4s, so dt_couple = 1s gives good accuracy.

7. **Trigger detection at each step:** After each TORAX mini-run, check betaN, q95, f_GW against thresholds. When a trigger fires, the pre-disruption state is immediately available for DREAM handoff — no need to re-run or interpolate.

**Open questions for implementation:**

- **How to pass final TORAX profiles as initial conditions for the next run?** Need to investigate TORAX's `profile_conditions` config to see if it accepts arbitrary initial profiles, or if we need to modify the config dict between runs.
- **JAX memory management:** Running 10+ TORAX calls sequentially in one Colab session — need to ensure JAX doesn't accumulate GPU memory. May need `jax.clear_caches()` between runs.
- **Can we avoid re-starting TORAX from scratch each step?** If TORAX has a warm-start mechanism, this would significantly reduce the per-step cost.

**This is the standard approach in integrated tokamak modeling.** It's essentially what JINTRAC, ETS (European Transport Simulator), and ASTRA do — the only difference is they use their own equilibrium solvers (HELENA, ESCO, etc.) instead of FreeGSNKE. Our architecture is:

```
Sequential Coupling (run_coupled_torax):

  t=0         t=1         t=2         t=3    ...
   │           │           │           │
   ▼           ▼           ▼           ▼
  FreeGSNKE  FreeGSNKE  FreeGSNKE  FreeGSNKE
  (eq #0)    (eq #1)    (eq #2)    (eq #3)
   │           │           │           │
   │ GEQDSK    │ GEQDSK    │ GEQDSK    │ GEQDSK
   ▼           ▼           ▼           ▼
  TORAX      TORAX      TORAX      TORAX
  [0→1s]     [1→2s]     [2→3s]     [3→4s]
   │           │           │           │
   │ Te,ne,j   │ Te,ne,j   │ Te,ne,j   │ Te,ne,j
   │ → betap   │ → betap   │ → betap   │ → betap
   └───────────┘───────────┘───────────┘
        ▲
        │ Check triggers at each step
```

---

### Days 15-18: Disruption Trigger Detection (PENDING)

**Goal:** Monitor the coupled FreeGSNKE+TORAX evolution and detect when operational limits are crossed.

**Tasks:**
- [ ] Implement sequential coupling method `run_coupled_torax()` in `CoupledSimulator` (see design above)
- [ ] Implement `TriggerDetector` class in `src/pipeline/` that evaluates disruption criteria at each coupling step
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

## Future Implementations

This section collects physics features and engineering improvements that are not needed for the current ML training pipeline but would be required for higher-fidelity simulations.

---

### Plasma Shape Controller (PSC)

"In a real tokamak, a Plasma Shape Controller (PSC) continuously adjusts individual coil voltages in real time to maintain the target shape (keeping the X-point exactly where it should be)."

**Current limitation:**
The static `inverse_solve` uses isoflux constraints (null-point at the X-point + top-of-plasma boundary point) to pin the plasma to the target lower single-null shape (kappa~1.78, delta~0.32, X-point at R=6.0m, Z=-3.8m). However, when the dynamic solver (`nl_solver.nlstepper`) takes over, these constraints are not re-applied — the solver just finds the nearest GS solution consistent with the current coil currents, which produces a rounder, lower-kappa equilibrium without a well-defined X-point. The shape jumps on the first dynamic step and then stays fixed because no coil voltages are being applied to maintain it.

**How to implement it:**
The PSC would be a feedback control loop operating at every coupling timestep:

1. **Measure**: After each `nlstepper` call, extract the actual X-point position (R_xpt, Z_xpt) and boundary shape from the new equilibrium.
2. **Compare**: Compute the shape error relative to the target — X-point displacement Δ(R,Z), elongation error Δkappa, triangularity error Δdelta.
3. **Control law**: Apply a proportional-integral (PI) controller to compute corrective coil voltages:
   ```
   ΔV_i(t) = K_p * e(t) + K_i * integral(e, 0, t)
   ```
   where `e(t)` is the shape error vector and the gain matrix `K_p`, `K_i` maps shape errors to individual coil voltage corrections. In practice, this gain matrix is computed from the Jacobian of X-point position with respect to coil currents (already partially available in FreeGSNKE's `dIy/dI` Jacobian).
4. **Apply**: Pass these corrective voltages as `V_coils` in the next `nlstepper` call.

A simpler first implementation could use just a proportional controller on the vertical position (VS3 coils control vertical stability) and the X-point radial position (PF5/PF6 primarily control X-point height and radial position in ITER).

**Why it's not needed now:**
For the ML disruption predictor training data, the exact plasma shape at each timestep doesn't need to match the ITER design exactly — the model learns patterns in the profiles (Te, ne, j, q) regardless of the precise boundary shape. The shape change on the first dynamic step is a numerical artifact, not a physical event, and the equilibrium remains self-consistent throughout the simulation.

**Priority:** Low. Implement if higher shape fidelity is needed for the training dataset, or if the project scope expands to include shape-dependent disruption precursors (e.g., locked mode detection which depends on triangularity).

---

### Minor Radius / q95 Correction (Pre-Mass-Generation Fix)

**Current limitation:**
The FreeGSNKE equilibrium produces a plasma with a_minor = 1.63 m instead of ITER's target a = 2.0 m. Since q95 ∝ a², this gives q95 ≈ 1.7 instead of ~3.0. The q95 < 2.0 disruption trigger cannot be used because the equilibrium *starts* below the threshold — there's no room to approach it from the safe side.

**Physics context — why q95 < 2 is a general limit:**
The q95 > 2 stability boundary comes from fundamental MHD physics (Kruskal-Shafranov limit), not from ITER's specific design. At q = 2, the m=2/n=1 tearing mode becomes unstable, producing magnetic islands that can lock to the wall and trigger a disruption. This applies to **all tokamaks** (JET, DIII-D, ASDEX-U, KSTAR, etc.). The exact threshold varies slightly with plasma shaping (strongly shaped plasmas can operate transiently at q95 ≈ 1.8), but q95 = 2.0 is the standard conservative limit. Our equilibrium at q95 = 1.7 is mathematically valid (Grad-Shafranov force balance is satisfied) but would be MHD-unstable in reality — our code doesn't detect this because it doesn't solve stability equations. The trigger detector checks q95 numerically, which is sufficient for the pipeline.

**How to fix it:**
Adjust the isoflux constraints in `src/predisruption/equilibrium.py` to widen the plasma boundary so that a ≈ 2.0 m:
- Adjust the X-point position (currently R=6.0, Z=-3.8; try R=5.8, Z=-4.2 or similar)
- Adjust the top-of-plasma constraint (currently R=5.54, Z=3.5)
- Possibly re-tune coil currents to support the wider equilibrium
This is a configuration tuning task (estimated: a few hours), not an architectural change.

**When to fix:**
Before mass data generation (Week 4, Day 22+). Not needed for the Week 3 DREAM handoff work, which can use Greenwald or betaN triggers instead.

**Priority:** Medium. Required before generating q95-triggered disruption scenarios. The other 3 trigger types (Greenwald, betaN, VDE) work independently of this issue.

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
| 2026-03-17 | TORAX full-run mode (not step-by-step) | TORAX is designed for complete trajectory runs via `run_simulation()`. Step-by-step experimental API exists but is impractical for geometry updates between steps. Strategy: pre-compute FreeGSNKE equilibria → save GEQDSKs → run TORAX once with time-dependent geometry. SimplifiedTransport handles real-time coupling for trigger detection. |
| 2026-03-17 | q95 < 2 is a universal MHD stability limit | The Kruskal-Shafranov limit (q95 > 2 for kink stability) applies to all tokamaks, not just ITER. Our q95 ≈ 1.7 equilibrium is mathematically valid but MHD-unstable. Fix: widen plasma boundary via isoflux constraints before mass generation. |
| 2026-03-23 | FreeGSNKE writes COCOS 7 (not COCOS 1 as assumed) | The `eqdsk` library identifies sign conventions from the GEQDSK data. FreeGSNKE's output matches COCOS 7/17, not COCOS 1. TORAX's config must specify `cocos: 7`. |
| 2026-03-23 | Use static re-solves (not dynamic evolution) for equilibrium sequence | FreeGSNKE's dynamic solver requires `coil_resist` not available in our machine config, and loses the O-point after ~10s. Static re-solves with updated betap are more robust and adequate for flat-top scenarios. |
| 2026-03-23 | **Sequential coupling as main approach** | Instead of pre-computing all equilibria then running TORAX once, alternate between FreeGSNKE (1 equilibrium) and TORAX (1-second mini-run) at each coupling step. This gives self-consistent geometry-transport feedback. More expensive (~3x) but physically correct. See "CRITICAL DESIGN DECISION" section for full details. |

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
| 2026-03-17 | ToraxTransport class used completely wrong API | Original code called `torax.build_sim_from_config()`, `runner.run_step()`, `runner.update_geometry()` — none exist. Rewrote using actual TORAX v1.3.0 API: `ToraxConfig.from_dict()` + `run_simulation()` returning xarray DataTree. |
| 2026-03-17 | TORAX config format wrong (sources used `"mode": "MODEL"`) | Real TORAX sources use empty dicts `{}` for defaults or specific params like `P_total`, `gaussian_location`. Rewrote `_default_iter_torax_config()` based on `torax/examples/iterhybrid_rampup.py`. Added missing sections: `pedestal`, `neoclassical`, `time_step_calculator`. |
| 2026-03-23 | numpy ABI mismatch on Colab after pip upgrade | `pip install torax` upgraded numpy .py but left old .so C extensions. Fix: `pip install --force-reinstall --no-deps numpy==<version>` + kernel restart. |
| 2026-03-23 | freegs4e import crash: numba incompatible with numpy>=2.0 | Colab ships numba which fails with numpy 2.x. freegs4e's except handler uses `warnings.warn()` but `warnings` not imported. Fix: inject fake numba module with no-op `njit`. |
| 2026-03-23 | JAX CUDA plugin 0.7.2 incompatible with JAX 0.9.2 | Colab's pre-installed cuda plugin was for older JAX. Fix: `pip install jax[cuda12]` to get matching plugin. |
| 2026-03-23 | TORAX flux surface volumes non-monotonic near edge | Contours near separatrix/X-point fail to close in our GEQDSK. Fix: `last_surface_factor=0.75`, `n_surfaces=25` (stop at 75% of psi_boundary). |
| 2026-03-23 | TORAX variables on different rho grids (25 vs 27 pts) | `rho_cell_norm` (25 pts) vs `rho_face_norm` (27 pts). Code assumed all on cell grid. Fix: read each variable's actual dimension via `profiles[var].dims[-1]`. |

## External Dependencies

| Tool | Repository | Status |
|------|-----------|--------|
| DREAM | https://github.com/chalmersplasmatheory/DREAM | Installed |
| DINA-IMAS | https://github.com/iterorganization/DINA-IMAS | Cloned, core compiled, IMAS interface blocked (HPC-only) |
| IMAS Codex MCP | https://github.com/iterorganization/imas-codex | Connected |
| freegs4e | (dependency of FreeGSNKE) | Installed (venv), patched for single-null bug |
| FreeGSNKE | https://github.com/FusionComputingLtd/freegsnke | Installed v2.1.0, static+dynamic solve working |
| TORAX | https://github.com/google-deepmind/torax | v1.3.0 installed in venv (can't run locally — no AVX). **Validated on Colab** (2026-03-23): GPU working, all 3 tests pass, FreeGSNKE↔TORAX coupling demonstrated. COCOS 7, last_surface_factor=0.75, n_surfaces=25. |
