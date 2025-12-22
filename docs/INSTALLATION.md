# Installation Guide

This document provides detailed instructions for installing the Tokamak Disruption Simulator and its dependencies.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Installation](#basic-installation)
3. [DREAM Installation](#dream-installation)
4. [DINA-IMAS Installation](#dina-imas-installation)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- Operating System: Linux (Ubuntu 20.04+ recommended) or macOS
- RAM: 8 GB minimum, 16 GB recommended
- Disk Space: 5 GB for codes and dependencies, additional space for datasets
- GPU: Not required (CPU-only simulations)

### Required Software

- Python 3.10 or higher
- Git
- CMake 3.12 or higher
- C++17 compatible compiler (GCC 7.0+ or Clang 5.0+)
- GNU Make

### Required Libraries

For DREAM compilation:
- GNU Scientific Library (GSL) 2.4 or higher
- HDF5 library
- PETSc (Portable, Extensible Toolkit for Scientific Computation)

## Basic Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/username/tokamak-disruption-simulator.git
cd tokamak-disruption-simulator
```

### Step 2: Create Python Environment

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate tokamak_sim
```

Using pip and venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Basic Installation

```bash
python -c "import numpy; import scipy; import h5py; print('Basic installation successful')"
```

## DREAM Installation

DREAM (Disruption Runaway Electron Analysis Model) is required for simulating disruption physics.

### Step 1: Install System Dependencies

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libgsl-dev \
    libhdf5-dev \
    libopenmpi-dev \
    git
```

On macOS (using Homebrew):

```bash
brew install cmake gsl hdf5 open-mpi
```

### Step 2: Install PETSc

```bash
# Download PETSc
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc

# Configure PETSc (adjust paths as needed)
./configure \
    --with-cc=gcc \
    --with-cxx=g++ \
    --with-fc=gfortran \
    --with-debugging=0 \
    --with-shared-libraries=1 \
    COPTFLAGS="-O3" \
    CXXOPTFLAGS="-O3" \
    FOPTFLAGS="-O3"

# Build PETSc
make all
make check

# Set environment variables (add to ~/.bashrc)
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-linux-c-opt  # or appropriate architecture name
```

### Step 3: Install DREAM

```bash
# Clone DREAM repository
git clone https://github.com/chalmersplasmatheory/DREAM.git
cd DREAM

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. \
    -DPETSC_DIR=$PETSC_DIR \
    -DPETSC_ARCH=$PETSC_ARCH

# Build DREAM
make -j4

# Add DREAM Python interface to path (add to ~/.bashrc)
export PYTHONPATH="/path/to/DREAM/py:$PYTHONPATH"
export PATH="/path/to/DREAM/build:$PATH"
```

### Step 4: Verify DREAM Installation

```bash
# Test DREAM executable
dreami --help

# Test Python interface
python -c "import DREAM; print('DREAM Python interface available')"
```

## DINA-IMAS Installation

DINA-IMAS is used for simulating plasma scenario evolution.

### Step 1: Clone DINA-IMAS Repository

```bash
git clone https://github.com/iterorganization/DINA-IMAS.git
cd DINA-IMAS
```

### Step 2: Configure Environment

```bash
source ci_header.sh
```

### Step 3: Build DINA

```bash
make
```

### Step 4: Verify Installation

```bash
# Test DINA GUI (requires display)
python tools/GUI/main.py

# Run example scenario
cd machines/ITER
# Follow instructions in README for running scenarios
```

## Verification

After completing installation, run the verification script:

```bash
python scripts/verify_installation.py
```

This script checks:
- Python dependencies
- DREAM availability (optional)
- DINA availability (optional)
- Directory structure integrity

## Troubleshooting

### Common Issues

#### PETSc Configuration Fails

Ensure you have all required compilers:
```bash
which gcc g++ gfortran
```

#### DREAM CMake Cannot Find PETSc

Verify environment variables:
```bash
echo $PETSC_DIR
echo $PETSC_ARCH
```

#### Python Import Errors for DREAM

Check PYTHONPATH includes DREAM py directory:
```bash
echo $PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"
```

#### HDF5 Version Mismatch

Ensure consistent HDF5 versions:
```bash
h5cc -showconfig | grep "HDF5 Version"
python -c "import h5py; print(h5py.version.hdf5_version)"
```

### Getting Help

- DREAM documentation: https://ft.nephy.chalmers.se/dream
- DINA-IMAS issues: https://github.com/iterorganization/DINA-IMAS/issues
- Project issues: https://github.com/username/tokamak-disruption-simulator/issues
