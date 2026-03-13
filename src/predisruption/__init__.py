"""
Pre-disruption plasma evolution module.

Couples FreeGSNKE (free-boundary MHD equilibrium) with TORAX (1D transport)
to simulate plasma evolution from startup to disruption trigger.

When TORAX/JAX is unavailable (no AVX CPU support), falls back to a
simplified NumPy transport model that provides physics-plausible profiles.
"""

from .iter_machine import build_iter_machine
from .equilibrium import EquilibriumSolver
from .transport import TransportSolver
from .coupling import CoupledSimulator
from .shot_runner import ShotRunner

__all__ = [
    "build_iter_machine",
    "EquilibriumSolver",
    "TransportSolver",
    "CoupledSimulator",
    "ShotRunner",
]
