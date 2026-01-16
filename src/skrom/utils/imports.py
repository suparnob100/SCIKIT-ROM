"""
The `utils` package aggregates frequently used libraries and helper routines for the pyHyperRom framework:
  – Core imports for OS, filesystem, timing, and randomness
  – Numerical & symbolic computing: NumPy, SciPy (dense & sparse), Sympy
  – Finite‐element tools: scikit‐fem (`from skfem import *`), mesh I/O via meshio
  – Linear algebra solvers: dense (LU), sparse (splu, spilu, CG), and multigrid (pyamg)
  – Sampling & design‐of‐experiments: Sobol, Latin Hypercube (SciPy QMC & pyDOE)
  – Plotting & visualization: Matplotlib (2D/3D, custom styles), ptitprince, and optional animation modules  
  – Utilities for dynamic imports, path handling, and Cartesian products
These imports and utilities ensure consistent, ready‐to‐use functionality across the entire reduced‐order modeling pipeline.
"""

# Import standard Python libraries for system interaction and time handling
import os  # Operating system utilities
import sys  # System-specific parameters and functions
import time  # Time-related functions
import random  # Random number generation
from importlib import reload  # Dynamic module reloading

# Modify system path to include the directory of this script for local imports
# desired_path = os.path.join(os.path.dirname(__file__))  # Determine the directory of the current file
# sys.path.append(desired_path)  # Append the directory to the system path

import numpy as np  # Array operations and numerical computing
import sympy as sp  # Symbolic mathematics
from scipy import sparse  # Sparse matrix data structures
from scipy.sparse import linalg  # Sparse linear algebra
from itertools import product  # Cartesian product generator

from skfem import *  # Finite element assembly and solving
from scipy.linalg import lu_factor, lu_solve, norm  # Dense linear algebra routines

import matplotlib  # Plotting library
import matplotlib.pyplot as plt  # MATLAB-like plotting interface
from matplotlib import cm  # Colormap functions
from matplotlib.ticker import LinearLocator  # Tick locator for axes
from matplotlib.ticker import MaxNLocator  # Max number of ticks locator
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting tools
# import sci_mplstyle_package  # Custom matplotlib styles
from pyamg import smoothed_aggregation_solver  # Algebraic multigrid solver
from scipy.sparse import csc_matrix  # Compressed sparse column matrix format
from scipy.sparse.linalg import splu, spilu, LinearOperator, cg  # Sparse direct and iterative solvers
import meshio  # Mesh input/output library

from scipy.stats.qmc import Sobol  # Sobol sequence sampler
from scipy.stats.qmc import LatinHypercube  # Latin Hypercube sampler
import importlib  # Import utilities
from pyDOE import lhs  # Latin Hypercube design of experiments
from pathlib import Path  # Filesystem path abstraction
from scipy.linalg import qr  # QR decomposition
from scipy.sparse import issparse, csr_matrix

def _petsc_is_available() -> bool:
    try:
        import petsc4py  # noqa: F401
        from petsc4py import PETSc  # noqa: F401
        return True
    except Exception:
        return False

print(f"PETSc available: {_petsc_is_available()}")

# Apply a custom plotting style for publication-quality figures
#plt.style.use(os.path.join(desired_path, 'utils/plot_files/publication.mplstyle'))

# Import custom animation modules for 1D and 2D data visualization
#from utils.plot_files.animate_1D import AnimatedPlot  # Handles 1D animated plotting
#from utils.plot_files.animate_2D import AnimatedContourPlot  # Handles 2D contour animations

# Optional imports (uncomment if needed)
# from pylab import plot  # Provides MATLAB-like plotting functions
# import fnnlsEigen as fe # Implements non-negative least squares (NNLS) using Eigen
# import decompyle3  # Tool for decompiling Python bytecode