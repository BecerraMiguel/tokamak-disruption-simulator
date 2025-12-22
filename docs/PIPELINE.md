# Pipeline Architecture

This document describes the data generation pipeline architecture and workflow.

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Components](#pipeline-components)
3. [Data Flow](#data-flow)
4. [Handoff Mechanism](#handoff-mechanism)
5. [Configuration](#configuration)

## Overview

The tokamak disruption simulator generates synthetic plasma shot data by coupling two physics codes:

1. **DINA**: Simulates plasma evolution during normal operation and approach to disruption
2. **DREAM**: Simulates the disruption itself (Thermal Quench and Current Quench)

The pipeline handles both disruptive and non-disruptive shots:

- **Non-disruptive shots**: DINA only (full scenario simulation)
- **Disruptive shots**: DINA (pre-disruption) + DREAM (disruption)

## Pipeline Components

### Scenario Generator

Generates randomized plasma scenarios within specified parameter ranges:
- Plasma current variations
- Density profiles
- Heating power levels
- Initial conditions

### DINA Runner

Executes DINA simulations with configured scenarios:
- Solves Grad-Shafranov equilibrium
- Evolves transport equations
- Monitors operational limits

### Trigger Detector

Monitors plasma parameters for disruption conditions:
- Greenwald fraction: `n_e / n_G > threshold`
- Beta limit: `beta_N > threshold`
- Safety factor: `q_95 < threshold`

### Handoff Module

Transfers plasma state from DINA to DREAM:
- Extracts radial profiles at trigger time
- Interpolates to DREAM grid
- Configures DREAM initial conditions

### DREAM Runner

Executes DREAM simulations for disruption phase:
- Thermal Quench with anomalous transport
- Current Quench with resistive decay
- Optional runaway electron physics

### Signal Combiner

Merges DINA and DREAM outputs:
- Temporal alignment
- Signal continuity verification
- Resampling to uniform frequency

### Dataset Writer

Writes generated shots to HDF5 format:
- Signal arrays with metadata
- Labels and trigger information
- Dataset-level statistics

## Data Flow

```
[Scenario Config] --> [DINA Runner] --> [Non-disruptive shot]
                           |
                           v
                  [Trigger Detector]
                           |
                           v (if triggered)
                    [Handoff Module]
                           |
                           v
                    [DREAM Runner]
                           |
                           v
                  [Signal Combiner] --> [Disruptive shot]
                           |
                           v
                  [Dataset Writer] --> [HDF5 Dataset]
```

## Handoff Mechanism

The handoff from DINA to DREAM occurs at the trigger time `t_trigger`:

### Data Transferred

| Parameter | From DINA | To DREAM |
|-----------|-----------|----------|
| T_e(r) | Temperature profile | Initial electron temperature |
| n_e(r) | Density profile | Ion species density |
| j(r) | Current profile | Current distribution |
| I_p | Total current | Normalization |
| Geometry | R_0, a, kappa, delta | Radial grid setup |

### Continuity Requirements

- All signals must be continuous at `t_trigger`
- No discontinuities in physical quantities
- Smooth transition of derivatives where possible

## Configuration

Pipeline behavior is controlled through YAML configuration files:

```yaml
generation:
  n_shots_total: 1000
  disruptive_fraction: 0.5
  random_seed: 42

dina:
  scenario_type: "ITER_15MA"
  duration: 10.0  # seconds

dream:
  thermal_quench_time: 1.0e-3  # seconds
  current_quench_time: 50.0e-3  # seconds

output:
  sampling_frequency: 10000  # Hz
  format: "hdf5"
```

See `configs/generation.yaml` for full configuration options.
