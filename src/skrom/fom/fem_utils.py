"""
Finite element utilities for preconditioning, mesh operations, and nonlinear solving.

This module includes:
- `build_pc_amgsa`: construct AMG smoothed aggregation preconditioner.
- `element2location`: generate element-to-coordinate mappings from mesh data.
- `newton_solver` and its helpers: perform Newton–Raphson solves with or without Dirichlet BC.
- `compute_basis_regions`: create reduced bases for mesh regions.
- `load_domain`, `load_mesh_and_basis`: domain/mesh loading utilities.
- `unwrap_attr`: attribute unwrapping helper.

Author: Suparno Bhattacharyya
"""
import numpy as np
from skrom.utils.imports import *
from pyamg import smoothed_aggregation_solver
from skfem import solve


def build_pc_amgsa(A, **kwargs):
    """
    Build an algebraic multigrid smoothed aggregation preconditioner.

    **TL;DR**: Creates an AMG preconditioner from a system matrix for efficient 
    iterative solving of large sparse linear systems.

    Parameters
    ----------
    A : scipy.sparse matrix or array_like
        The system matrix for which the preconditioner is constructed.
    **kwargs : dict
        Additional keyword arguments passed to pyamg.smoothed_aggregation_solver.

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        The preconditioner as a linear operator suitable for use in iterative 
        solvers like conjugate gradient.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(100, 100))
    >>> M = build_pc_amgsa(A)
    >>> # Use M as preconditioner in iterative solver
    """
    return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()


def element2location(mesh):
    """
    Map mesh elements to their spatial coordinates.

    **TL;DR**: Extracts element-wise coordinate information from mesh connectivity,
    useful for element-based computations in finite element methods.

    Parameters
    ----------
    mesh : object
        Mesh object with attributes `p` (node coordinates) and `t` (element 
        connectivity). Expected to have `p` as shape (spatial_dim, n_nodes) 
        and `t` as shape (n_local_nodes_per_element, n_elements).

    Returns
    -------
    element_coords : ndarray of shape (n_elements, n_local_nodes)
        Array of element coordinates, where each row corresponds to an element 
        and each column corresponds to a local node within the element. For 1D 
        meshes, this gives coordinates of element endpoints.

    Examples
    --------
    >>> # For a 1D mesh with 3 elements and 4 nodes
    >>> element_coords = element2location(mesh)
    >>> # element_coords[0] gives coordinates of first element's nodes
    """
    # Access node coordinates
    node_coords = mesh.p  # (spatial_dim, n_nodes)

    # Access element connectivity
    element_nodes = mesh.t  # (n_local_nodes_per_element, n_elements)

    # Get element-to-location mapping
    element_coords = node_coords[:, element_nodes]  # (spatial_dim, n_local_nodes, n_elements)

    # Reshape for clarity (for 1D)
    element_coords = element_coords.squeeze().T  # (n_elements, n_local_nodes)

    return element_coords


def compute_basis_regions(basis, masks):
    """
    Create reduced basis functions for specified mesh regions.

    **TL;DR**: Given boolean masks defining mesh regions, returns basis functions
    restricted to each region for efficient regional computations.

    Parameters
    ----------
    basis : object
        Finite element basis object with `nelems` attribute and `with_elements` method.
    masks : dict of str to ndarray of bool
        Dictionary mapping region names to boolean element masks of shape 
        (basis.nelems,). True values indicate elements belonging to the region.

    Returns
    -------
    region_bases : dict of str to object
        Dictionary mapping region names to reduced basis objects containing only 
        elements specified by the corresponding mask.

    Examples
    --------
    >>> masks = {'left': np.array([True, False, True]), 
    ...          'right': np.array([False, True, False])}
    >>> region_bases = compute_basis_regions(basis, masks)
    >>> # region_bases['left'] contains basis for left region elements
    """
    elem_indices = np.arange(basis.nelems)
    return {
        name: basis.with_elements(elem_indices[mask])
        for name, mask in masks.items()
    }


def load_domain(instance):
    """
    Load domain information and assign attributes to instance.

    **TL;DR**: Calls instance.domain() and assigns all returned attributes 
    to the instance object for convenient access.

    Parameters
    ----------
    instance : object
        Object with a `domain()` method that returns a dictionary of domain 
        attributes.

    Notes
    -----
    This function modifies the instance in-place by setting attributes based 
    on the domain dictionary keys and values.

    Examples
    --------
    >>> load_domain(problem_instance)
    >>> # Now problem_instance.mesh, problem_instance.basis, etc. are available
    """
    dom = instance.domain()
    for name, val in dom.items():
        setattr(instance, name, val)


def load_mesh_and_basis(instance):
    """
    Load only mesh and basis from domain and assign to instance.

    **TL;DR**: Extracts just mesh and basis from instance.domain() for cases 
    where only these two components are needed.

    Parameters
    ----------
    instance : object
        Object with a `domain()` method returning a dictionary containing 
        at least 'mesh' and 'basis' keys.

    Notes
    -----
    This function modifies the instance in-place by setting only `mesh` and 
    `basis` attributes, ignoring other domain components.

    Examples
    --------
    >>> load_mesh_and_basis(problem_instance)
    >>> # Now problem_instance.mesh and problem_instance.basis are available
    """
    # grab only the first two returned values
    dom = instance.domain()
    mesh = dom["mesh"]
    basis = dom["basis"]

    # automatically set the attributes
    for name, val in zip(("mesh", "basis"), (mesh, basis)):
        setattr(instance, name, val)


def unwrap_attr(instance, attr_name: str):
    """
    Unwrap 0-dimensional object arrays to their scalar values.

    **TL;DR**: Converts 0-d numpy object arrays to their contained scalar value
    using .item(), useful for cleaning up attributes after certain operations.

    Parameters
    ----------
    instance : object
        Object containing the attribute to unwrap.
    attr_name : str
        Name of the attribute to unwrap.

    Notes
    -----
    Only applies unwrapping if the attribute is a 0-dimensional numpy array 
    with object dtype. The instance is modified in-place.

    Examples
    --------
    >>> # If instance.result is np.array(42, dtype=object)
    >>> unwrap_attr(instance, 'result')
    >>> # Now instance.result is 42 (scalar int)
    """
    val = getattr(instance, attr_name, None)
    if isinstance(val, np.ndarray) and val.dtype == object and val.shape == ():
        setattr(instance, attr_name, val.item())


def newton_solver(
    assemble_fn,
    rhs_fn, 
    u0: np.ndarray,
    dirichlet_dofs: np.ndarray | None = None,
    dirichlet_vals: np.ndarray | None = None,
    *assemble_args,
    tol: float = 1e-2,
    maxit: int = 50,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Solve a nonlinear system using the Newton–Raphson method.

    **TL;DR**: Newton solver with optional Dirichlet boundary conditions that 
    iteratively solves F(u) = 0 by computing Jacobian and RHS at each iteration.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the Jacobian matrix. Should accept (u, *assemble_args) 
        and return the Jacobian matrix J where J[i,j] = ∂F_i/∂u_j.
    rhs_fn : callable
        Function that assembles the RHS vector. Should accept (u, *assemble_args) 
        and return the residual vector F(u).
    u0 : ndarray of shape (n_dofs,)
        Initial guess for the solution vector.
    dirichlet_dofs : array_like of int, optional
        Indices of degrees of freedom with prescribed Dirichlet boundary conditions.
        If None or empty, no Dirichlet BC are applied.
    dirichlet_vals : ndarray, optional
        Prescribed values at the Dirichlet DOFs. Must have same length as 
        `dirichlet_dofs`.
    *assemble_args : tuple
        Additional positional arguments passed to `assemble_fn` and `rhs_fn`.
    tol : float, default=1e-2
        Convergence tolerance based on the L2 norm of the solution update.
    maxit : int, default=50
        Maximum number of Newton iterations before giving up.
    alpha : float, default=1.0
        Step length parameter for solution updates.

    Returns
    -------
    u : ndarray of shape (n_dofs,)
        Approximate solution vector after convergence or maximum iterations.

    Raises
    ------
    RuntimeError
        If the solver fails to converge within `maxit` iterations and no 
        Dirichlet boundary conditions are specified.

    Notes
    -----
    The Newton method solves F(u) = 0 by iterating u_{k+1} = u_k - α * J_k^{-1} * F(u_k),
    where J_k is the Jacobian at u_k. For problems with Dirichlet BC, the system 
    is condensed to solve only for free DOFs.
    """
    # normalize BC inputs
    if dirichlet_dofs is None or len(dirichlet_dofs) == 0:
        return _newton_no_dirichlet_bc(assemble_fn, rhs_fn, u0, *assemble_args, 
                                       tol=tol, maxit=maxit, alpha=alpha)
    else:
        return _newton_with_dirichlet_bc(
            assemble_fn, rhs_fn, u0,
            dirichlet_dofs, dirichlet_vals,
            *assemble_args, tol=tol, maxit=maxit, alpha=alpha)


def _newton_with_dirichlet_bc(assemble_fn, rhs_fn, u0, dofs, vals, *args, 
                              tol, maxit, jac_conditioner=False, alpha=1.0):
    """
    Internal Newton–Raphson solver for systems with Dirichlet boundary conditions.

    **TL;DR**: Newton solver that enforces Dirichlet BC by condensing the system
    to solve only for free DOFs, with optional AMG preconditioning.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the Jacobian matrix given solution vector.
    rhs_fn : callable  
        Function that assembles the RHS/residual vector given solution vector.
    u0 : ndarray
        Initial guess for the solution vector.
    dofs : array_like of int
        Indices of Dirichlet degrees of freedom.
    vals : ndarray
        Prescribed values at the Dirichlet DOFs.
    *args : tuple
        Additional positional arguments passed to assembly functions.
    tol : float
        Convergence tolerance based on solution update norm.
    maxit : int
        Maximum number of Newton iterations.
    jac_conditioner : bool, default=False
        Whether to use AMG preconditioning based on the Jacobian matrix.
    alpha : float, default=1.0
        Step length parameter, automatically reduced if convergence stalls.

    Returns
    -------
    u : ndarray
        Solution vector satisfying Dirichlet BC after convergence.

    Notes
    -----
    Uses system condensation to eliminate Dirichlet DOFs from the linear system.
    If `jac_conditioner=True`, builds an AMG preconditioner for the condensed system.
    Step length `alpha` is halved every 40 iterations to aid convergence.
    """
    u = u0.copy()
    pc = None
    u[dofs] = vals

    free_dofs = np.setdiff1d(np.arange(len(u)), dofs)
    for itr in range(maxit):
        J = assemble_fn(u, *args) 
        RHS = rhs_fn(u, *args)
        u_prev = u.copy()
        
        if jac_conditioner:
            Jc, RHSc, _, free = condense(J, RHS, D=dofs)
            if itr < 2:
                pc = build_pc_amgsa(Jc)
            delta = solve(
                Jc, -RHSc,
                solver=solver_iter_pcg(verbose=False, M=pc, tol=1e-16, atol=1e-16)
            )
            u += delta
        else:
            if (itr + 1) % 40 == 0:
                alpha *= 0.5
                print(f"Iteration {itr}: reducing α → {alpha}")

            u += alpha * solve(*condense(J, -RHS, I=free_dofs))
            
            # u += alpha*solve(*condense(J, -RHS, D=dofs))

        if np.linalg.norm(u - u_prev) < tol:
            return u
        elif itr == maxit - 1:
            print("Newton solver did not converge!")
            return u


def _newton_no_dirichlet_bc(assemble_fn, rhs_fn, u0, *args, tol, maxit, alpha=1.0):
    """
    Internal Newton–Raphson solver for systems without Dirichlet boundary conditions.

    **TL;DR**: Standard Newton solver with AMG preconditioning for unconstrained
    nonlinear systems, automatically building preconditioner in first iterations.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the Jacobian matrix given solution vector.
    rhs_fn : callable
        Function that assembles the RHS/residual vector given solution vector.  
    u0 : ndarray
        Initial guess for the solution vector.
    *args : tuple
        Additional positional arguments passed to assembly functions.
    tol : float
        Convergence tolerance based on solution update norm.
    maxit : int
        Maximum number of Newton iterations.
    alpha : float, default=1.0
        Step length parameter (currently unused in this variant).

    Returns
    -------
    u : ndarray
        Solution vector after convergence.

    Raises
    ------
    RuntimeError
        If convergence is not achieved within `maxit` iterations.

    Notes
    -----
    Builds an AMG preconditioner from the Jacobian in the first two iterations
    for efficient solution of the linear systems. Uses preconditioned conjugate 
    gradient with tight tolerances for the linear solves.
    """
    u = u0.copy()
    pc = None
    for itr in range(maxit):
        J, RHS = assemble_fn(u, *args), rhs_fn(u, *args)
        u_prev = u.copy()
        if itr < 2:
            pc = build_pc_amgsa(J)
        delta = solve(
            J, -RHS,
            solver=solver_iter_pcg(verbose=False, M=pc, tol=1e-16, atol=1e-16)
        )
        u += delta
        if np.linalg.norm(u - u_prev) < tol:
            return u
    raise RuntimeError("Newton solver did not converge!")
