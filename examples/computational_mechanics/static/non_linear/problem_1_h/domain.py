import numpy as np
from skfem import MeshHex, Basis, ElementVector, ElementHex1
from skrom.fom.fem_utils import element2location, compute_basis_regions


def domain_(lx = 10.0, ly = 1.0, lz = 1.0, factor = 6 ):
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

    nx, ny, nz = (factor * int(l) for l in (lx, ly, lz))

    # 2) Create a structured hexahedral mesh
    mesh = MeshHex.init_tensor(
        np.linspace(0, lx, nx + 1),
        np.linspace(0, ly, ny + 1),
        np.linspace(0, lz, nz + 1),
    ).with_boundaries({
        'left':   lambda x: np.isclose(x[0], 0.0),
        'right':  lambda x: np.isclose(x[0], lx),
    })

    element = ElementVector(ElementHex1())
    basis = Basis(mesh, element, intorder=1)

    right_dofs = basis.get_dofs('right')
    dirichlet_dofs = basis.get_dofs({'right', 'left'})

    free_dofs = basis.complement_dofs(dirichlet_dofs)

    # 6) Return all domain components as a dict
    return {
        'mesh': mesh,
        'basis': basis,
        'right_dofs': right_dofs,
        'dirichlet_dofs': dirichlet_dofs,
        'free_dofs': free_dofs,
    }
