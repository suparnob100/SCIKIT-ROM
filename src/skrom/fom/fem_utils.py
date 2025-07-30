"""
This module includes:
- `build_pc_amgsa`: construct AMG smoothed aggregation preconditioner.
- `element2location`: generate element-to-coordinate mappings from mesh data.
- `newton_solver` and its helpers: perform Newton–Raphson solves with or without Dirichlet BC.

[Author: Suparno Bhattacharyya]
"""
import numpy as np
from skrom.utils.imports import *
from pyamg import smoothed_aggregation_solver
from skfem import solve

def build_pc_amgsa(A, **kwargs):
    """
    Build an algebraic multigrid smoothed aggregation preconditioner.

    Parameters
    ----------
    A : scipy.sparse matrix or array_like
        The system matrix for which the preconditioner is constructed.
    **kwargs :
        Additional keyword arguments passed to pyamg.smoothed_aggregation_solver.

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        The preconditioner as a linear operator suitable for use in iterative solvers.
    """
    return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()

def element2location(mesh):
    """
    Map mesh elements to their spatial coordinates.

    Parameters
    ----------
    mesh : object
        Mesh object with attributes `p` (node coordinates) and `t` (element connectivity).

    Returns
    -------
    element_coords : ndarray of shape (n_elements, n_local_nodes)
        Array of element coordinates, where each row corresponds to an element and
        each column corresponds to a local node within the element.
    """
    # Access node coordinates
    node_coords = mesh.p  # (1, n_nodes)

    # Access element connectivity
    element_nodes = mesh.t  # (n_local_nodes_per_element, n_elements)

    # Get element-to-location mapping
    element_coords = node_coords[:, element_nodes]  # (1, n_local_nodes, n_elements)

    # Reshape for clarity (for 1D)
    element_coords = element_coords.squeeze().T  # (n_elements, n_local_nodes)

    return element_coords

def compute_basis_regions(basis, masks):
    """
    Given a dict mapping region names to boolean element‐masks,
    returns a dict of reduced bases for each region.
    """
    elem_indices = np.arange(basis.nelems)
    return {
        name: basis.with_elements(elem_indices[mask])
        for name, mask in masks.items()
    }

def load_domain(instance):
    dom = instance.domain()
    for name, val in dom.items():
        setattr(instance, name, val)

def load_mesh_and_basis(instance):
    """
    Call domain() and assign just mesh and basis onto self.
    """
    # grab only the first two returned values
    dom = instance.domain()
    mesh  = dom["mesh"]
    basis = dom["basis"]

    # automatically set the attributes
    for name, val in zip(("mesh", "basis"), (mesh, basis)):
        setattr(instance, name, val)

def unwrap_attr(instance, attr_name: str):
        """
        If self.<attr_name> is a 0-d object ndarray, replace it with its .item().
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
    alpha = 1.0
) -> np.ndarray:
    """
    Solve a nonlinear system using the Newton–Raphson method with optional Dirichlet boundary conditions.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the system. Should return either
        (RHS, J) for functions with dirichlet BC or (J, RHS) for no-dirichlet variant.
    u0 : ndarray
        Initial guess for the solution vector.
    dirichlet_dofs : array_like of int, optional
        Indices of degrees of freedom with prescribed Dirichlet boundary conditions.
        If None or empty, no Dirichlet BC are applied.
    dirichlet_vals : ndarray, optional
        Values at the Dirichlet DOFs.
    *assemble_args
        Additional positional arguments passed to `assemble_fn`.
    tol : float, default 1e-2
        Tolerance for convergence based on the norm of the update.
    maxit : int, default 50
        Maximum number of Newton iterations.

    Returns
    -------
    u : ndarray
        Approximate solution vector after convergence.

    Raises
    ------
    RuntimeError
        If the solver fails to converge within `maxit` iterations.
    """
    # normalize BC inputs
    if dirichlet_dofs is None or len(dirichlet_dofs) == 0:
        return _newton_no_dirichlet_bc(assemble_fn, rhs_fn, u0, *assemble_args, tol=tol, maxit=maxit,alpha=alpha)
    else:
        return _newton_with_dirichlet_bc(
            assemble_fn, rhs_fn, u0,
            dirichlet_dofs, dirichlet_vals,
            *assemble_args, tol=tol, maxit=maxit,alpha=alpha)

def _newton_with_dirichlet_bc(assemble_fn, rhs_fn, u0, dofs, vals, *args, tol, maxit, jac_conditioner = False, alpha = 1.0):
    """
    Internal Newton–Raphson solver for systems with Dirichlet boundary conditions.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the system, returning (RHS, J) for given `u`.
    u0 : ndarray
        Initial guess for the solution vector.
    dofs : array_like of int
        Indices of Dirichlet degrees of freedom.
    vals : ndarray
        Prescribed values at the Dirichlet DOFs.
    *args
        Additional positional arguments passed to `assemble_fn`.
    tol : float
        Tolerance for convergence based on the norm of the update.
    maxit : int
        Maximum number of Newton iterations.
    jac_conditioner : bool, default False
        Whether to use a Jacobian-based preconditioner for linear solves.

    Returns
    -------
    u : ndarray
        Solution vector satisfying Dirichlet BC after convergence.

    Raises
    ------
    RuntimeError
        If convergence is not achieved within `maxit` iterations.
    """
    u = u0.copy()
    pc = None
    u[dofs] = vals

    free_dofs = np.setdiff1d(np.arange(len(u)), dofs)
    for itr in range(maxit):
        J =  assemble_fn(u, *args) 
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
            u+=delta
        else:
            if (itr+1 % 40 == 0):
                alpha *= 0.5
                print(f"Iteration {itr}: reducing α → {alpha}")

            u+= alpha*solve(*condense(J, -RHS, I=free_dofs))
            
            # u += alpha*solve(*condense(J, -RHS, D=dofs))

        if np.linalg.norm(u - u_prev) < tol:
            return u
        elif itr == maxit - 1:
            print("Newton solver did not converge!")
            return u

def _newton_no_dirichlet_bc(assemble_fn, rhs_fn, u0, *args, tol, maxit):
    """
    Internal Newton–Raphson solver for systems without Dirichlet boundary conditions.

    Parameters
    ----------
    assemble_fn : callable
        Function that assembles the system, returning (J, RHS) for given `u`.
    u0 : ndarray
        Initial guess for the solution vector.
    *args
        Additional positional arguments passed to `assemble_fn`.
    tol : float
        Tolerance for convergence based on the norm of the update.
    maxit : int
        Maximum number of Newton iterations.

    Returns
    -------
    u : ndarray
        Solution vector after convergence.

    Raises
    ------
    RuntimeError
        If convergence is not achieved within `maxit` iterations.
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
