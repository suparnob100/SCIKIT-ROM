import numpy as np
from skfem import MeshLine, Basis, ElementLineP1

"""
Construct 1-D FEM domain for heat conduction with Dirichlet boundary.

Parameters
----------
rx : float
    Domain length [0, rx].
factor : int
    Mesh refinement exponent; number of elements = 2**factor.
left_bc : float
    Left boundary coordinate (unused).
right_bc : float
    Coordinate where Dirichlet BC is applied.
dirichlet_boundary_value : float
    Value imposed at Dirichlet DOFs.
dirichlet_boundary_point : float
    Identifier for the boundary point (unused).

Returns
-------
dict
    {
        'basis': Basis,
        'free_dofs': ndarray,
        'dirichlet_dofs': ndarray,
        'dirichlet_boundary_value': float,
        'mesh': MeshLine
    }
"""

def domain_(
    rx=0.5,
    factor=17,
    left_bc=0.0,
    right_bc=0.5,
    dirichlet_boundary_value=573.15,
    dirichlet_boundary_point=0.5,
):
    # compute number of elements and nodes
    nx = 2 ** factor
    mesh = MeshLine(np.linspace(0, rx, nx + 1))  # uniform 1D mesh

    # create linear (P1) finite-element basis
    element = ElementLineP1()
    basis = Basis(mesh, element)

    # identify Dirichlet DOFs where x == right_bc
    dirichlet_boundary_dofs = basis.get_dofs(
        lambda x: np.isclose(x[0], right_bc)
    )
    # compute free DOFs (all others)
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)

    # return domain components as a dict
    return {
        'basis': basis,
        'free_dofs': free_dofs,
        'dirichlet_boundary_dofs': dirichlet_boundary_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value,
        'mesh': mesh
    }