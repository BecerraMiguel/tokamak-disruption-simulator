# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tokamak Disruption Simulator: a synthetic data generation pipeline for tokamak plasma disruptions. Couples two external physics codes (DINA for pre-disruption evolution, DREAM for disruption physics) to produce labeled HDF5 datasets for training ML disruption predictors (particularly Fourier Neural Operators). A phenomenological model serves as fallback when DINA/DREAM are unavailable.

## Build & Environment Commands

```bash
# Environment setup (conda)
conda env create -f environment.yml
conda activate tokamak_sim

# Environment setup (pip)
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
bash check_dependencies.sh

# Run tests
pytest tests/
pytest tests/test_pipeline.py              # single test file
pytest tests/test_pipeline.py -k test_name # single test

# Code quality
black src/
flake8 src/
mypy src/

# Generate dataset
python scripts/generate_dataset.py --config configs/generation.yaml
```

## Architecture

**Data flow:** Scenario Config → DINA Runner → Trigger Detector → Handoff → DREAM Runner → Signal Combiner → HDF5 Writer

Four modules under `src/`:
- `dina/` — DINA interface: scenario setup, execution, signal extraction
- `dream/` — DREAM interface: configuration, execution, signal extraction
- `pipeline/` — Orchestration: trigger detection, DINA→DREAM handoff, signal combining, dataset generation
- `utils/` — Shared: physics constants, HDF5 I/O, visualization

**Configuration:** YAML-driven (`configs/generation.yaml`). DINA and DREAM each have sub-configs under `configs/dina/` and `configs/dream/`.

**Disruption triggers** (operational limits in `generation.yaml`):
- Greenwald fraction > 0.95
- Beta_N > 3.2
- q95 < 2.2

**Output:** HDF5 files under `data/` with structure `shots/shot_NNNNN/{signals/*, time, label}` + `metadata/`.

## Key Conventions

- Python 3.10+, formatted with Black, linted with flake8, typed with mypy
- Classes use CamelCase, functions/modules use snake_case
- Physical quantities carry SI units in comments/docs
- DINA and DREAM are optional system dependencies (C++/CMake/PETSc/GSL); the phenomenological model runs without them

## Progress Tracking

After completing all tasks for a given day in the implementation plan (`Plan_Implementacion_Simulacion_Tokamak.pdf`), **always update `PROJECT_PROGRESS.md`** with:
- Summary of what was done
- Important decisions made and their rationale
- Problems encountered and their solutions
- What should be done next (upcoming day's tasks)

This ensures continuity across sessions.

## MCP Servers

- **IMAS Codex** (`imas-codex`): Connected to `https://imas-dd.iter.org/mcp`. Provides access to IMAS Data Dictionary — use it for querying IDS schemas, understanding IMAS data paths, and physics context when working with DINA-IMAS or DREAM's IMAS interface.
