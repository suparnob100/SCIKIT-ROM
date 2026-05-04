"""Template domain definition.

TL;DR
-----
This module shows where a user should create the mesh, basis, and boundary degree-of-freedom sets.

Notes
-----
The returned dictionary is expected by the problem workflow classes.
"""

def domain_(**kwargs):
    """Template for setting up the computational domain and boundary conditions.
    
    TL;DR
    -----
    Template for setting up the computational domain and boundary conditions.
    
    Parameters
    ----------
    kwargs : dict
        Any problem-specific parameters needed for mesh generation,
        basis construction, and boundary identification.
    
    Returns
    -------
    dict
        {
            'mesh': <mesh object>,
            'basis': <finite-element basis object>,
            'free_dofs': <array of free DOFs>,
            'dirichlet_dofs': <array of Dirichlet DOFs>,
            'dirichlet_value': <value imposed on Dirichlet BC>,
            # add other entries as needed
        }
    """
    # TODO: generate mesh
    # TODO: build finite-element basis
    # TODO: identify boundary DOFs
    # TODO: compute free DOFs

    return {
        'mesh': mesh,
        'basis': basis,
        'free_dofs': free_dofs,
        'dirichlet_dofs': dirichlet_dofs,
        'dirichlet_value': dirichlet_value,
    }