"""
The `rom` folder provides core reduced‐order modeling tools:

  – Projection & assembly classes (`BilinearFormROM`, `LinearFormROM`):
      • Project full‐order stiffness and load contributions onto reduced bases
      • Handle Dirichlet boundary conditions via free‐DOF mappings
      • Support chunked/clustered assembly for memory‐efficient reduced operators

  – Error evaluation & visualization (`error_utils.py`):
      • Compute time‐dependent and global error metrics (L2, RMSE, R², energy‐norm, etc.)
      • Generate scatter plots, spatial snapshots, raincloud plots, and tabular reports

  – ROM utilities (`utils.py`):
      • Snapshot splitting and sampling (random, Latin‐Hypercube, Sobol, Gaussian)
      • Basis enrichment via deflation and QR re‐orthonormalization
      • Solution reconstruction from reduced coefficients
      • Data I/O for ROM simulations (`rom_data_gen`, `load_rom_data`)
      • Newton solvers for (hyper‐)reduced systems
"""

from .ecm import (
    dense_ecm_weights,
    flat_to_element_gauss_weights,
    active_ecm_elements,
    BilinearFormHYPERROM_ecm,
    LinearFormHYPERROM_ecm,
)
