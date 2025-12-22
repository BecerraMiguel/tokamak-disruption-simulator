# API Reference

This document provides the API reference for the tokamak disruption simulator.

## Table of Contents

1. [src.dina](#srcdina)
2. [src.dream](#srcdream)
3. [src.pipeline](#srcpipeline)
4. [src.utils](#srcutils)

---

## src.dina

### scenario.py

Module for configuring DINA plasma scenarios.

#### Classes

##### `DinaScenario`

Configuration container for a DINA simulation scenario.

```python
class DinaScenario:
    """
    DINA scenario configuration.
    
    Attributes:
        device (str): Tokamak device name (e.g., "ITER", "DIII-D")
        ip (float): Target plasma current [A]
        duration (float): Scenario duration [s]
        heating_power (float): Auxiliary heating power [W]
        density_target (float): Target line-averaged density [m^-3]
    """
```

### runner.py

Module for executing DINA simulations.

#### Functions

##### `run_dina_scenario`

```python
def run_dina_scenario(
    scenario: DinaScenario,
    output_dir: str,
    verbose: bool = False
) -> DinaOutput:
    """
    Execute a DINA simulation with the specified scenario.
    
    Args:
        scenario: DINA scenario configuration
        output_dir: Directory for output files
        verbose: Enable verbose logging
        
    Returns:
        DinaOutput object containing simulation results
        
    Raises:
        DinaExecutionError: If DINA execution fails
    """
```

### extractor.py

Module for extracting signals from DINA outputs.

#### Functions

##### `extract_signals`

```python
def extract_signals(
    dina_output: DinaOutput,
    signals: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Extract time-series signals from DINA output.
    
    Args:
        dina_output: DINA simulation output
        signals: List of signal names to extract
                 Default: ["ip", "beta_n", "q95", "n_e", "li"]
                 
    Returns:
        Dictionary mapping signal names to numpy arrays
    """
```

---

## src.dream

### configurator.py

Module for configuring DREAM simulations.

#### Classes

##### `DreamConfiguration`

```python
class DreamConfiguration:
    """
    DREAM simulation configuration.
    
    Attributes:
        radial_grid: Radial grid specification
        initial_profiles: Initial plasma profiles
        thermal_quench: Thermal quench parameters
        current_quench: Current quench parameters
    """
```

#### Functions

##### `create_dream_settings`

```python
def create_dream_settings(
    config: DreamConfiguration
) -> DREAMSettings:
    """
    Create a DREAM settings object from configuration.
    
    Args:
        config: DREAM configuration object
        
    Returns:
        DREAMSettings object ready for simulation
    """
```

### runner.py

Module for executing DREAM simulations.

#### Functions

##### `run_dream_simulation`

```python
def run_dream_simulation(
    settings: DREAMSettings,
    output_file: str
) -> DreamOutput:
    """
    Execute a DREAM simulation.
    
    Args:
        settings: DREAM settings object
        output_file: Path for output HDF5 file
        
    Returns:
        DreamOutput object containing simulation results
    """
```

---

## src.pipeline

### handoff.py

Module for DINA to DREAM state transfer.

#### Functions

##### `perform_handoff`

```python
def perform_handoff(
    dina_output: DinaOutput,
    trigger_time: float,
    dream_grid: RadialGrid
) -> DreamConfiguration:
    """
    Transfer plasma state from DINA to DREAM at trigger time.
    
    Args:
        dina_output: DINA simulation output
        trigger_time: Time of disruption trigger [s]
        dream_grid: Target DREAM radial grid
        
    Returns:
        DreamConfiguration with initial conditions from DINA
    """
```

### detector.py

Module for disruption trigger detection.

#### Functions

##### `detect_trigger`

```python
def detect_trigger(
    signals: Dict[str, np.ndarray],
    time: np.ndarray,
    limits: OperationalLimits
) -> Optional[float]:
    """
    Detect disruption trigger time from plasma signals.
    
    Args:
        signals: Dictionary of plasma signals
        time: Time array [s]
        limits: Operational limit thresholds
        
    Returns:
        Trigger time [s] if disruption detected, None otherwise
    """
```

### generator.py

Module for high-level dataset generation.

#### Classes

##### `DataGenerationPipeline`

```python
class DataGenerationPipeline:
    """
    High-level pipeline for generating tokamak disruption datasets.
    
    Methods:
        generate_non_disruptive_shots: Generate normal shots
        generate_disruptive_shots: Generate disruptive shots
        generate_dataset: Generate complete mixed dataset
    """
```

---

## src.utils

### physics.py

Physical constants and operational limits.

#### Constants

```python
# Fundamental constants
ELECTRON_MASS = 9.109e-31  # kg
ELECTRON_CHARGE = 1.602e-19  # C

# ITER parameters
ITER_MAJOR_RADIUS = 6.2  # m
ITER_MINOR_RADIUS = 2.0  # m
ITER_TOROIDAL_FIELD = 5.3  # T
ITER_PLASMA_CURRENT = 15.0e6  # A
```

#### Classes

##### `OperationalLimits`

```python
class OperationalLimits:
    """
    Tokamak operational limits for disruption detection.
    
    Attributes:
        greenwald_fraction (float): Maximum n_e/n_G (default: 1.0)
        beta_n_limit (float): Maximum beta_N (default: 3.5)
        q95_minimum (float): Minimum q_95 (default: 2.0)
    """
```

### io.py

Input/output utilities.

#### Functions

##### `save_dataset`

```python
def save_dataset(
    shots: List[Shot],
    filepath: str,
    metadata: Dict = None
) -> None:
    """
    Save generated shots to HDF5 dataset.
    
    Args:
        shots: List of Shot objects
        filepath: Output HDF5 file path
        metadata: Optional dataset metadata
    """
```

##### `load_dataset`

```python
def load_dataset(
    filepath: str
) -> Tuple[List[Shot], Dict]:
    """
    Load shots from HDF5 dataset.
    
    Args:
        filepath: Input HDF5 file path
        
    Returns:
        Tuple of (shots list, metadata dict)
    """
```

### visualization.py

Plotting utilities.

#### Functions

##### `plot_shot`

```python
def plot_shot(
    shot: Shot,
    signals: List[str] = None,
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    Plot time-series signals for a single shot.
    
    Args:
        shot: Shot object to plot
        signals: List of signals to include
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
```
