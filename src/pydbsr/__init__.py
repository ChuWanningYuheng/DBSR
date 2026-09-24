"""pydbsr: Python interface to the DBSR Dirac B-spline R-matrix codes (O. Zatsarinny).

Building blocks
---------------
Ion                 target atom / ion (``Ion('Xe', 1)``)
Target              DHF target states (dbsr_hf) -> jj states
Scattering          inner region: dbsr_prep3, conf3, breit3, mat3, hd3 (parallel)
OuterRegion         K-matrices, collision strengths, cross sections, rates
nist                NIST ASD levels -> experimental thresholds
run                 run any DBSR program directly
"""
from .atoms import Ion
from .io import Grid, partial_waves, to_dbsr_conf
from .outer import CollisionStrengths, OuterRegion, read_h
from .runner import DBSRError, available_programs, bin_dir, run, run_partial_waves
from .scattering import Scattering
from .structure import State, Target
from . import nist

__version__ = "0.1.0"

__all__ = ["Ion", "Target", "State", "Scattering", "OuterRegion", "CollisionStrengths", "read_h",
           "nist", "run", "run_partial_waves", "DBSRError", "bin_dir", "available_programs",
           "Grid", "partial_waves", "to_dbsr_conf"]
