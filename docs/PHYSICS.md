# Physics Background

This document describes the physics models and concepts underlying the tokamak disruption simulator.

## Table of Contents

1. [Introduction to Tokamak Disruptions](#introduction-to-tokamak-disruptions)
2. [Operational Limits](#operational-limits)
3. [Disruption Phases](#disruption-phases)
4. [DINA Physics Models](#dina-physics-models)
5. [DREAM Physics Models](#dream-physics-models)
6. [References](#references)

## Introduction to Tokamak Disruptions

A disruption is a catastrophic loss of plasma confinement in a tokamak, occurring on millisecond timescales. During a disruption, the plasma's thermal and magnetic energy is rapidly released to the surrounding structures, posing significant engineering challenges for fusion reactor design.

Disruptions are the primary operational safety concern for large tokamaks such as ITER, where unmitigated disruptions could cause severe damage to plasma-facing components and structural elements.

## Operational Limits

Disruptions typically occur when the plasma violates one or more operational limits:

### Greenwald Density Limit

The Greenwald limit defines the maximum line-averaged electron density:

```
n_G = I_p / (pi * a^2)  [10^20 m^-3]
```

Where:
- `I_p`: Plasma current [MA]
- `a`: Minor radius [m]

Operating above `n_e / n_G > 1.0` typically leads to disruption.

### Troyon Beta Limit

The normalized beta is limited by MHD stability:

```
beta_N = beta_t * a * B_0 / I_p < 3.5
```

Where:
- `beta_t`: Toroidal beta [%]
- `B_0`: Toroidal magnetic field [T]

### Safety Factor Limit

The edge safety factor must remain above a critical value:

```
q_95 > 2.0
```

Operating below this limit leads to global MHD instabilities.

## Disruption Phases

### Precursor Phase (100-300 ms)

- Growing MHD modes
- Approach to operational limits
- Degrading confinement

### Thermal Quench (0.1-2 ms)

- Rapid loss of thermal energy
- Electron temperature drops from ~10 keV to ~10 eV
- Caused by stochastic magnetic field lines

### Current Quench (10-100 ms)

- Decay of plasma current due to increased resistivity
- Generation of intense induced electric fields
- Potential runaway electron generation

## DINA Physics Models

[To be completed - will describe Grad-Shafranov solver, transport equations, etc.]

## DREAM Physics Models

[To be completed - will describe thermal quench model, current diffusion, runaway physics]

## References

1. Hender, T.C. et al. (2007). "MHD stability, operational limits and disruptions." Nuclear Fusion 47, S128.

2. de Vries, P.C. et al. (2011). "Survey of disruption causes at JET." Nuclear Fusion 51, 053018.

3. Hoppe, M. et al. (2021). "DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations." Computer Physics Communications 268, 108098.

4. ITER Physics Basis (1999). Nuclear Fusion 39, 2137.
