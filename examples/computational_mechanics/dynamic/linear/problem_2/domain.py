"""
Domain setup for a 1-D, two-segment Timoshenko beam (cantilever).

The beam occupies x in [0, L] and is split into two regions:
- region_1: elements with centroid x <= split
- region_2: elements with centroid x  > split

Unknown fields (per x):
- w(x): transverse displacement
- theta(x): cross-section rotation

Dirichlet BC (cantilever root):
- w(0) = 0
- theta(0) = 0
"""

import numpy as np
from skfem import MeshLine, Basis, ElementLineP1
from skrom.fom.fem_utils import element2location, compute_basis_regions


def domain_(
    L: float = 1.0,
    n_elem: int = 25600,
    split: float = 0.5,
):
    """
    Build mesh, basis, boundary DOFs, and region bases.

    Parameters
    ----------
    L : float
        Beam length.
    n_elem : int
        Number of line elements.
    split : float
        Region split location in x.

    Returns
    -------
    dict
        Mesh/basis and bookkeeping for ROM workflows.
    """
    # Mesh and scalar FE basis (P1)
    mesh = MeshLine(np.linspace(0.0, L, n_elem + 1))
    element = ElementLineP1()
    basis = Basis(mesh, element)

    # Dirichlet DOF on the left boundary (x == 0)
    left_dofs_view = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
    left_dofs = left_dofs_view.all()

    # DOF on the right boundary (x == L), used for sign convention
    right_dofs_view = basis.get_dofs(lambda x: np.isclose(x[0], L))
    right_dofs = right_dofs_view.all()

    free_dofs = basis.complement_dofs(left_dofs_view)

    # Region masks (based on element node coordinates)
    element_x = element2location(mesh)              # (nelems, 2) in 1D
    x_centroid = element_x.mean(axis=1)             # (nelems,)
    regions_mask = {
        "region_1": x_centroid <= split,
        "region_2": x_centroid > split,
    }
    basis_regions = compute_basis_regions(basis, regions_mask)
    global_mask = tuple(regions_mask.items())

    return {
        "mesh": mesh,
        "basis": basis,                      # scalar basis (shared by w and theta)
        "basis_regions": basis_regions,
        "global_mask": global_mask,
        "L": L,
        "n_scalar_dofs": basis.N,
        "dirichlet_boundary_dofs_scalar": left_dofs,   # ndarray[int]
        "dirichlet_boundary_dofs": left_dofs_view,     # DofsView (kept for compatibility)
        "free_dofs_scalar": free_dofs,
        "right_boundary_dofs_scalar": right_dofs,      # ndarray[int]
        "dirichlet_boundary_value": 0.0,
    }
