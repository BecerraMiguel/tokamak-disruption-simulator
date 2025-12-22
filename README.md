# Tokamak Disruption Simulator

A high-fidelity synthetic data generation system for tokamak plasma disruptions using DINA and DREAM physics codes.

## Overview

This project provides a pipeline for generating realistic synthetic datasets of tokamak plasma shots, including both normal (non-disruptive) and disruptive scenarios. The generated data is intended for training machine learning models, particularly Fourier Neural Operators (FNO), for disruption prediction.

### Key Features

- Integration with DINA plasma scenario simulator for pre-disruption evolution
- Integration with DREAM for disruption physics (Thermal Quench and Current Quench)
- Automated handoff between DINA and DREAM at disruption trigger time
- Configurable scenarios with parametric variations
- HDF5 output format compatible with ML training pipelines

## Project Structure

```
tokamak-disruption-simulator/
├── configs/              # Configuration files for DINA and DREAM
│   ├── dina/
│   └── dream/
├── src/                  # Source code
│   ├── dina/             # DINA interface and utilities
│   ├── dream/            # DREAM interface and utilities
│   ├── pipeline/         # Data generation pipeline
│   └── utils/            # Common utilities
├── scripts/              # Executable scripts
├── notebooks/            # Jupyter notebooks for analysis
├── docs/                 # Documentation
├── data/                 # Generated datasets
├── results/              # Results and figures
└── tests/                # Unit tests
```

## Requirements

### System Requirements

- Python 3.10 or higher
- CMake 3.12 or higher (for DREAM compilation)
- C++17 compatible compiler (gcc 7.0 or higher)
- GNU Scientific Library (GSL) 2.4 or higher
- HDF5 library
- PETSc (for DREAM)

### Python Dependencies

See `requirements.txt` for the complete list of Python dependencies.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/tokamak-disruption-simulator.git
cd tokamak-disruption-simulator
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate tokamak_sim
```

Alternatively, using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install DREAM (Optional)

For full disruption physics simulation, install DREAM following the instructions in `docs/INSTALLATION.md`.

### 4. Install DINA-IMAS (Optional)

For realistic plasma scenario simulation, install DINA-IMAS following the instructions in `docs/INSTALLATION.md`.

## Usage

### Quick Start

```bash
# Generate a synthetic dataset with default configuration
python scripts/generate_dataset.py --config configs/generation.yaml

# Validate the generated dataset
python scripts/validate_dataset.py --input data/tokamak_disruption_dataset.h5
```

### Configuration

Dataset generation can be configured through YAML files. See `configs/generation.yaml` for available options.

## Dataset Format

The generated dataset is stored in HDF5 format with the following structure:

```
tokamak_disruption_dataset.h5
├── shots/
│   ├── shot_00001/
│   │   ├── signals/
│   │   │   ├── ip          # Plasma current [A]
│   │   │   ├── beta_n      # Normalized beta
│   │   │   ├── q95         # Safety factor at 95% flux
│   │   │   ├── n_e         # Electron density [m^-3]
│   │   │   └── li          # Internal inductance
│   │   ├── time            # Time array [s]
│   │   └── label           # 0: normal, 1: disruptive
│   └── ...
└── metadata/
    ├── n_shots
    ├── sampling_frequency
    └── generation_date
```

## Documentation

- `docs/INSTALLATION.md` - Detailed installation instructions
- `docs/PHYSICS.md` - Physics background and model descriptions
- `docs/PIPELINE.md` - Pipeline architecture and data flow
- `docs/API.md` - API reference

## References

### Physics Codes

- DREAM: Hoppe, Embreus, Fulop (2021). "DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations." Computer Physics Communications 268, 108098.
- DINA-IMAS: https://github.com/iterorganization/DINA-IMAS

### Disruption Physics

- Hender et al. (2007). "MHD stability, operational limits and disruptions." Nuclear Fusion 47.
- de Vries et al. (2011). "Survey of disruption causes at JET." Nuclear Fusion 51.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing

Contributions are welcome. Please read the contributing guidelines before submitting pull requests.

## Contact

For questions or issues, please open an issue on the GitHub repository.
