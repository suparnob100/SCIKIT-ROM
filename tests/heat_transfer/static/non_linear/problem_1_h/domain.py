import numpy as np
from skfem import MeshLine, Basis, ElementLineP1
from src.skrom.fom.fem_utils import element2location, compute_basis_regions


def domain_(
    rx=0.5,
    factor=17,
    left_bc=0.0,
    right_bc=0.5,
    dirichlet_boundary_value=573.15,
    dirichlet_boundary_point=0.5,
):
    """
    Construct 1-D FEM domain for heat conduction.

    Parameters
    ----------
    rx : float
        Length of the interval [0, rx].
    factor : int
        Mesh refinement exponent; number of elements = 2**factor.
    left_bc : float
        Left boundary coordinate (unused for Dirichlet here).
    right_bc : float
        Coordinate where Dirichlet BC is applied.
    dirichlet_boundary_value : float
        Value to impose at Dirichlet DOFs.
    dirichlet_boundary_point : float
        Identifier for BC point (returned for reference).

    Returns
    -------
    dict
        {
          'mesh': MeshLine,
          'basis': Basis,
          'free_dofs': ndarray,
          'dirichlet_boundary_dofs': ndarray,
          'dirichlet_boundary_value': float,
          'basis_regions': dict
        }
    """
    # 1) Generate uniform 1-D mesh
    nx = 2 ** factor
    mesh = MeshLine(np.linspace(0, rx, nx))

    # 2) Define linear (P1) finite-element basis
    element = ElementLineP1()
    basis = Basis(mesh, element)

    # 3) Identify Dirichlet boundary DOFs at x == right_bc
    dirichlet_boundary_dofs = basis.get_dofs(
        lambda x: np.isclose(x[0], right_bc)
    )
    # 4) Compute free DOFs
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)

    # 5) Partition elements into two regions by x = 0.4
    element2loc = element2location(mesh)
    break_point  = 0.4
    bool_arr    = element2loc < break_point
    regions_mask = {
        'region_1': np.all(bool_arr, axis=1),
        'region_2': ~np.all(bool_arr, axis=1),
    }
    global_mask = tuple(regions_mask.items())                    # convert to tuple for passing through the assemble function

    # 6) Return all domain components as a dict
    return {
        'nx' : nx,
        'mesh': mesh,
        'basis': basis,
        'free_dofs': free_dofs,
        'dirichlet_boundary_dofs': dirichlet_boundary_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value,
        'global_mask' : global_mask
    }
