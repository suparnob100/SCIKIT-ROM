import numpy as np
from skfem import MeshHex, Basis, ElementVector, ElementHex1
from skrom.fom.fem_utils import element2location, compute_basis_regions
import numpy as np
from skfem import MeshLine, Basis, ElementLineP1


import numpy as np
from skfem import MeshLine, ElementLineP1, Basis

def domain_(
    rx = 420.0,
    factor = 14,
    left_bc=0.0,
    right_bc=420.0,
    dirichlet_boundary_value=0.0,
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
    mesh = MeshLine(np.linspace(0.0, rx, nx + 1))

    # 2) Define linear (P1) finite-element basis
    element = ElementLineP1()
    basis = Basis(mesh, element, intorder = 4)

    # 3) Identify Dirichlet boundary DOFs at x == left_bc
    dirichlet_boundary_dofs = basis.get_dofs(
        lambda x: np.isclose(x[0], left_bc)
    )

    # 4) Compute free DOFs
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)


    return {
        'mesh': mesh,
        'basis': basis,
        'free_dofs': free_dofs,
        'dirichlet_boundary_dofs': dirichlet_boundary_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value,
    }