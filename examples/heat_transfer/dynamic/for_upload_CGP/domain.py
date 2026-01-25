# Domain dimensions and mesh resolution
import numpy as np
from skfem import MeshTet, Basis, ElementVector, FacetBasis, ElementTetP2
from skrom.fom.fem_utils import element2location, compute_basis_regions

def domain_3d(lx=40., ly=10., lz=6, factor=2, dirichlet_boundary_value=298.0):
    
    nx, ny, nz = (factor * int(l) for l in (lx, ly, lz))

    # # Create a tetrahedral mesh
    # mesh = MeshTet.init_tensor(
    #     np.linspace(0, lx, nx+1),
    #     np.linspace(0, ly, ny+1),
    #     np.linspace(0, lz, nz+1)
    # )

    mesh_path = 'box_centerline_top_refine.msh'
    mesh = MeshTet.load(mesh_path)

    # Define boundary regions
    boundaries = {
        'left':   lambda x: np.isclose(x[0], 0.0),
        'right':  lambda x: np.isclose(x[0], lx),
        'front':  lambda x: np.isclose(x[1], 0.0),
        'back':   lambda x: np.isclose(x[1], ly),
        'bottom': lambda x: np.isclose(x[2], 0.0),
        'top':    lambda x: np.isclose(x[2], lz),
    }

    # Assign boundaries to the mesh
    mesh = mesh.with_boundaries(boundaries)

    # Define a scalar finite element for the temperature field
    element = ElementTetP2()
    basis = Basis(mesh, element)

    # Define a FacetBasis over all non-Dirichlet boundaries:
    non_dirichlet = {'left', 'right', 'front', 'back', 'top'}
    fbasis_non_dirichlet = FacetBasis(mesh, element, facets=non_dirichlet, intorder=4*basis.elem.maxdeg)
    non_dirichlet_dofs = basis.get_dofs(non_dirichlet)

    fbasis_top = FacetBasis(mesh, element, facets='top') #,intorder=4)
    bottom_dofs = basis.get_dofs('bottom')
    free_dofs = np.setdiff1d(np.arange(basis.N), bottom_dofs)


    # 7) Return all domain data as a dict
    return {
        'mesh': mesh,
        'basis': basis,
        'nx': nx, 'ny': ny, 'nz': nz,
        'boundaries': mesh.boundaries,
        'element': element,
        'fbasis_non_dirichlet': fbasis_non_dirichlet,
        'fbasis_top': fbasis_top,
        'bottom_dofs': bottom_dofs,
        'free_dofs': free_dofs,
        'dirichlet_boundary_value': dirichlet_boundary_value
    }