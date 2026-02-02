"""
Data Generation Pipeline Module

This module implements the complete pipeline for generating synthetic
tokamak disruption data by coupling DINA and DREAM simulations.

Main components:
    - handoff: Transfer of plasma state from DINA to DREAM at disruption trigger
    - detector: Detection of disruption trigger conditions
    - combiner: Combination of DINA and DREAM outputs into unified signals
    - generator: High-level pipeline orchestration for dataset generation
"""