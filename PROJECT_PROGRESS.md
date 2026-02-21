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

### Day 5-6: DINA-IMAS Installation (IN PROGRESS)

**Date:** 2025-02-20
**Status:** Started - preliminary setup

**Tasks from plan:**
- [ ] Clone repository: https://github.com/iterorganization/DINA-IMAS.git
- [ ] Execute configuration script: source ci_header.sh
- [ ] Compile DINA: make
- [ ] Configure XML parameter files
- [ ] Test scenario preparation GUI (tools/GUI/main.py)
- [ ] Execute example scenario from machines/ folder

**Progress so far:**

#### IMAS Codex MCP Server Connected
- Discovered the IMAS Codex MCP server from an ITER presentation (14th International School on Integrated Modelling, June 2025)
- Repository: https://github.com/iterorganization/imas-codex (official ITER Organization repo)
- Connected Claude Code to the hosted IMAS MCP server at `https://imas-dd.iter.org/mcp`
- Command used: `claude mcp add --transport http imas-codex https://imas-dd.iter.org/mcp`
- This gives the AI agent direct access to the IMAS Data Dictionary with tools for:
  - Semantic search across IMAS data paths
  - Physics concept explanation within IMAS schemas
  - IDS (Interface Data Structure) analysis and exploration
  - Documentation search
- **Why this matters:** DINA-IMAS uses IMAS data structures heavily. Having direct access to the Data Dictionary will help correctly map DINA outputs to DREAM inputs (Te, ne, j, Ip, q profiles) and understand the IDS schema when configuring scenarios.

**Important decision:** Connected IMAS MCP before starting DINA installation to have better tooling support for understanding IMAS data structures throughout the process.

**Next steps (remaining for Day 5-6):**
- Clone and compile DINA-IMAS
- Configure environment and XML parameters
- Test with example scenarios
- Verify DINA produces expected outputs (Ip, betaN, q95, ne, Te, li, j profiles)

---

### Day 7: Integration & Verification (PENDING)

**Tasks:**
- Verify both codes can be imported from Python
- Create installation verification script
- Document installation process in docs/INSTALLATION.md
- Commit: "feat: complete installation of DINA and DREAM"

---

## WEEK 2: DINA Interface Module (PENDING)
*See plan pages 8-9 for details*

---

## WEEK 3: DREAM Interface & Handoff (PENDING)
*See plan pages 9-10 for details*

---

## WEEK 4: Pipeline Orchestration (PENDING)
*See plan pages 10-11 for details*

---

## WEEK 5: Dataset Generation & Validation (PENDING)
*See plan pages 11-12 for details*

---

## WEEK 6: Documentation & Delivery (PENDING)
*See plan pages 12-13 for details*

---

## Important Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-02-20 | Connect IMAS Codex MCP to Claude Code | Direct access to IMAS Data Dictionary improves understanding of DINA-IMAS data structures and IDS schemas, crucial for the DINA-DREAM handoff pipeline |

## Problems & Solutions Log

| Date | Problem | Solution |
|------|---------|----------|
| 2025-02-20 | Browser shows error when visiting MCP URL | Expected behavior - MCP requires SSE client (not browser). The `"Client must accept text/event-stream"` error confirms the server is running correctly |

## External Dependencies

| Tool | Repository | Status |
|------|-----------|--------|
| DREAM | https://github.com/chalmersplasmatheory/DREAM | Installed |
| DINA-IMAS | https://github.com/iterorganization/DINA-IMAS | Pending |
| IMAS Codex MCP | https://github.com/iterorganization/imas-codex | Connected |
