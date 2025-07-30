import numpy as np
import meshio
from skfem import MeshQuad, Basis, ElementQuad1, FacetBasis


def domain_():
    """
    Import a 2D triangular mesh and impose Dirichlet BC on all sides.

    Returns
    -------
    dict
        {
            'basis': Basis,
            'free_dofs': ndarray of free DOF indices,
            'dirichlet_boundary_dofs': ndarray of Dirichlet DOF indices,
            'dirichlet_boundary_value': float or tuple,
            'mesh': MeshQuad
        }
    """
    # mesh load and refinement
    mesh_path = 'star_outer10.msh'
    mesh = MeshQuad.load(mesh_path).refined(5)

    # element and basis
    element = ElementQuad1()
    basis   = Basis(mesh, element)

    # Dirichlet on all sides
    dirichlet = mesh.boundaries['boundary']
    fbasis_dirichlet = FacetBasis(mesh, element, facets=dirichlet)

    dirichlet_boundary_dofs = basis.get_dofs(dirichlet)
    free_dofs               = basis.complement_dofs(dirichlet_boundary_dofs)
    dirichlet_boundary_value = 0.0

    return {
        'mesh': mesh,
        'basis': basis,
        'free_dofs': free_dofs,
        'dirichlet_boundary_dofs': dirichlet_boundary_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value
    }

