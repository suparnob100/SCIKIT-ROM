import numpy as np
from skfem import MeshHex, Basis, ElementVector, FacetBasis, ElementHex1
from skrom.fom.fem_utils import element2location, compute_basis_regions

"""
Domain setup for 3-D linear elasticity: mesh, FE bases, and region partitioning.

Parameters
----------
lx, ly, lz : float, optional
    Physical dimensions of the block in x, y, z.
factor : int, optional
    Mesh resolution multiplier per unit length.
dirichlet_boundary_value : float, optional
    Dirichlet boundary displacement.

Returns
-------
dict
    Contains:
    - mesh: MeshHex
    - basis: Basis
    - nx, ny, nz: int
    - boundaries: dict of facet selectors
    - element: ElementVector
    - fbasis_dirichlet: FacetBasis
    - fbasis_neumann: FacetBasis
    - dirichlet_boundary_value: float
    - fbasis_faces: FacetBasis
    - dirichlet_dofs, neumann_dofs, faces_dofs: ndarray of int
    - basis_regions: dict of Basis objects per region
"""

def domain_(lx=10, ly=1, lz=1, factor=4, dirichlet_boundary_value=0.0):
    # 1) Compute mesh divisions
    nx, ny, nz = (factor * int(l) for l in (lx, ly, lz))

    # 2) Create a structured hexahedral mesh
    mesh = MeshHex.init_tensor(
        np.linspace(0, lx, nx + 1),
        np.linspace(0, ly, ny + 1),
        np.linspace(0, lz, nz + 1),
    ).with_boundaries({
        'left':   lambda x: np.isclose(x[0], 0.0),
        'right':  lambda x: np.isclose(x[0], lx),
        'front':  lambda x: np.isclose(x[1], 0.0),
        'back':   lambda x: np.isclose(x[1], ly),
        'bottom': lambda x: np.isclose(x[2], 0.0),
        'top':    lambda x: np.isclose(x[2], lz),
    })

    # 3) Build vector-valued FE basis
    element = ElementVector(ElementHex1())
    basis   = Basis(mesh, element)

    # 4) Facet bases for boundary conditions
    fbasis_dirichlet = FacetBasis(mesh, element, facets={'left'})
    fbasis_neumann   = FacetBasis(mesh, element, facets={'right'})
    fbasis_faces     = FacetBasis(mesh, element, facets={'top','bottom','front','back'})

    # 5) DOF indices for each boundary set
    dirichlet_dofs = basis.get_dofs({'left'})
    neumann_dofs   = basis.get_dofs({'right'})
    faces_dofs     = basis.get_dofs({'top','bottom','front','back'})

    # 6) Partition elements by x-coordinate
    break_point      = 0.5 * lx
    loc              = element2location(mesh)
    bool_arr_x       = loc[:,:,0] <= break_point
    regions_mask     = {'region_1': np.all(bool_arr_x, axis=1),
                        'region_2': ~np.all(bool_arr_x, axis=1)}
    basis_regions    = compute_basis_regions(basis, regions_mask)

    # 7) Return all domain data as a dict
    return {
        'mesh': mesh,
        'basis': basis,
        'nx': nx, 'ny': ny, 'nz': nz,
        'boundaries': mesh.boundaries,
        'element': element,
        'fbasis_dirichlet': fbasis_dirichlet,
        'fbasis_neumann': fbasis_neumann,
        'dirichlet_boundary_value': dirichlet_boundary_value,
        'fbasis_faces': fbasis_faces,
        'dirichlet_dofs': dirichlet_dofs,
        'neumann_dofs': neumann_dofs,
        'faces_dofs': faces_dofs,
        'basis_regions': basis_regions,
    }