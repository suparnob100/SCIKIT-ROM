"""\
Domain setup for 2-D transient heat conduction on a unit square.

The domain builder:
- creates a structured triangular mesh on [0, 1] x [0, 1]
- defines a P1 scalar basis for temperature
- identifies Dirichlet boundary DOFs (all outer boundaries)
- partitions elements into regions (single region by default)

Returned items follow the same keys used across the example folders.
"""

from __future__ import annotations

import numpy as np
from skfem import MeshTri, Basis, ElementTriP1

from skrom.fom.fem_utils import compute_basis_regions


def _structured_unit_square_mesh(nx: int, ny: int) -> MeshTri:
    """Create a structured triangulation of the unit square.

    This avoids reliance on MeshTri.init_* helpers (version differences).

    Parameters
    ----------
    nx, ny : int
        Number of rectangular cells in x and y.

    Returns
    -------
    MeshTri
        Triangular mesh with 2*nx*ny elements.
    """
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)

    # Node coordinates
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    p = np.vstack([X.ravel(), Y.ravel()])  # (2, npoints)

    # Connectivity
    def nid(i: int, j: int) -> int:
        return i + (nx + 1) * j

    tris = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i, j)
            n1 = nid(i + 1, j)
            n2 = nid(i, j + 1)
            n3 = nid(i + 1, j + 1)
            # two triangles per cell
            tris.append([n0, n1, n3])
            tris.append([n0, n3, n2])

    t = np.array(tris, dtype=np.int64).T  # (3, nelems)

    return MeshTri(p, t)


def domain_(
    nx: int = 25,
    ny: int = 25,
    dt: float = 2.0e-2,
    t_end: float = 4.0e-1,
    dirichlet_value: float = 0.0,
):
    """Construct a 2-D FEM domain for transient heat conduction.

    Parameters
    ----------
    nx, ny : int
        Number of rectangular cells in x and y (triangulated internally).
    dt : float
        Time step used by the FOM/ROM time integrators.
    t_end : float
        Final time.
    dirichlet_value : float
        Temperature value imposed on the boundary.

    Returns
    -------
    dict
        Keys follow the common scikit-rom example conventions.
    """
    mesh = _structured_unit_square_mesh(nx, ny)

    element = ElementTriP1()
    basis = Basis(mesh, element)

    # Dirichlet on the outer boundary of the unit square
    def on_boundary(x):
        return (
            np.isclose(x[0], 0.0) |
            np.isclose(x[0], 1.0) |
            np.isclose(x[1], 0.0) |
            np.isclose(x[1], 1.0)
        )

    dirichlet_boundary_dofs = basis.get_dofs(on_boundary)
    free_dofs = basis.complement_dofs(dirichlet_boundary_dofs)

    # Single-region partition (all elements)
    regions_mask = {"region_1": np.ones(basis.nelems, dtype=bool)}
    basis_regions = compute_basis_regions(basis, regions_mask)
    global_mask = tuple(regions_mask.items())

    nt = int(np.round(t_end / dt)) + 1
    t = np.linspace(0.0, t_end, nt)

    return {
        "mesh": mesh,
        "basis": basis,
        "free_dofs": free_dofs,
        "dirichlet_boundary_dofs": dirichlet_boundary_dofs,
        "dirichlet_boundary_value": float(dirichlet_value),
        "basis_regions": basis_regions,
        "global_mask": global_mask,
        "dt": float(dt),
        "t_end": float(t_end),
        "nt": int(nt),
        "t": t,
    }
