"""
Finite element utilities for preconditioning, mesh operations, and nonlinear solving.

This module provides utilities for finite element computations including AMG 
preconditioner construction, element-to-coordinate mappings, Newton-Raphson 
nonlinear solvers with Dirichlet boundary condition handling, regional basis 
computation, domain loading utilities, and attribute unwrapping helpers.

Notes
-----
**TL;DR**: Comprehensive toolkit for finite element method computations with 
focus on efficient iterative solving, mesh operations, and nonlinear problem 
handling.

Author
------
Suparno Bhattacharyya
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
        The system matrix for which the preconditioner is constructed
    **kwargs : dict
        Additional keyword arguments passed to pyamg.smoothed_aggregation_solver

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        The preconditioner as a linear operator suitable for use in iterative 
        solvers like conjugate gradient
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
        and `t` as shape (n_local_nodes_per_element, n_elements)

    Returns
    -------
    element_coords : ndarray of shape (n_elements, n_local_nodes)
        Array of element coordinates, where each row corresponds to an element 
        and each column corresponds to a local node within the element. For 1D 
        meshes, this gives coordinates of element endpoints
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
        Finite element basis object with `nelems` attribute and `with_elements` method
    masks : dict of str to ndarray of bool
        Dictionary mapping region names to boolean element masks of shape 
        (basis.nelems,). True values indicate elements belonging to the region

    Returns
    -------
    region_bases : dict of str to object
        Dictionary mapping region names to reduced basis objects containing only 
        elements specified by the corresponding mask
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
        attributes

    Notes
    -----
    This function modifies the instance in-place by setting attributes based 
    on the domain dictionary keys and values.
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
        at least 'mesh' and 'basis' keys

    Notes
    -----
    This function modifies the instance in-place by setting only `mesh` and 
    `basis` attributes, ignoring other domain components.
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
        Object containing the attribute to unwrap
    attr_name : str
        Name of the attribute to unwrap

    Notes
    -----
    Only applies unwrapping if the attribute is a 0-dimensional numpy array 
    with object dtype. The instance is modified in-place.
    """
    val = getattr(instance, attr_name, None)
    if isinstance(val, np.ndarray) and val.dtype == object and val.shape == ():
        setattr(instance, attr_name, val.item())



# ------------------------------------------------------------
# PETSc availability + SciPy CSR -> PETSc KSP solve (cached)
# ------------------------------------------------------------
def _petsc_is_available() -> bool:
    try:
        import petsc4py  # noqa: F401
        from petsc4py import PETSc  # noqa: F401
        return True
    except Exception:
        return False




def petsc_solve_csr(
    A_csr,
    b,
    *,
    ksp_type: str = "cg",
    pc_type: str = "ilu",
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_it: int = 2000,
    petsc_options: dict | None = None,
    cache: dict | None = None,
    ):

    """
    Solve A x = b using PETSc KSP, where A_csr is SciPy CSR (CPU).
    Uses a small cache to reuse PETSc Mat/KSP objects across calls.
    """
    from petsc4py import PETSc
    import scipy.sparse as sp

    if not sp.isspmatrix_csr(A_csr):
        A_csr = A_csr.tocsr()

    b = np.asarray(b, dtype=np.float64)
    n = A_csr.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"b has shape {b.shape}, expected ({n},)")

    # PETSc wants int32 indices/indptr for CSR setters on most builds
    indptr = np.asarray(A_csr.indptr, dtype=np.int32)
    indices = np.asarray(A_csr.indices, dtype=np.int32)
    data = np.asarray(A_csr.data, dtype=np.float64)

    # --- cache key: size + nnz + pointer arrays identity-ish ---
    # We assume the sparsity pattern is fixed for your FE assembly.
    key = (n, int(indptr[-1]))

    if cache is None:
        cache = {}

    state = cache.get("state", None)
    if state is None or state.get("key", None) != key:
        # (Re)build PETSc objects
        A = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(indptr, indices, data))
        A.setUp()
        A.assemble()

        x = PETSc.Vec().createSeq(n)
        bb = PETSc.Vec().createSeq(n)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)

        # Set KSP/PC basics
        ksp.setType(ksp_type)
        pc = ksp.getPC()
        pc.setType(pc_type)

        ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

        # Apply options database entries (optional)
        # These are the same keys you would pass via -ksp_* -pc_* etc.
        if petsc_options:
            opts = PETSc.Options()
            for kk, vv in petsc_options.items():
                if vv is None:
                    # allow toggles like {"ksp_monitor": None}
                    opts.setValue(kk, "")
                else:
                    opts.setValue(kk, str(vv))
            ksp.setFromOptions()
        else:
            ksp.setFromOptions()

        state = {"key": key, "A": A, "ksp": ksp, "x": x, "b": bb, "indptr": indptr, "indices": indices}
        cache["state"] = state

    else:
        # Reuse PETSc objects; only update numeric values.
        A = state["A"]
        ksp = state["ksp"]
        x = state["x"]
        bb = state["b"]

        # If pattern changed (should not in FE), rebuild to avoid PETSc "new nonzero" errors.
        if (state["indptr"].shape != indptr.shape) or (state["indices"].shape != indices.shape) \
           or (not np.array_equal(state["indptr"], indptr)) or (not np.array_equal(state["indices"], indices)):
            cache.pop("state", None)
            return petsc_solve_csr(
                A_csr, b,
                ksp_type=ksp_type, pc_type=pc_type,
                rtol=rtol, atol=atol, max_it=max_it,
                petsc_options=petsc_options, cache=cache,
            )

        A.zeroEntries()
        # This will fail if you try to insert a new nonzero -> but we guarded pattern.
        A.setValuesCSR(indptr, indices, data)
        A.assemble()

        # You can still update tolerances per-call if you want
        ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    bb.setArray(b)
    x.set(0.0)

    ksp.solve(bb, x)
    return x.getArray().copy()


# ------------------------------------------------------------
# Your previous SciPy-only Newton implementation (fallback)
# (kept as-is; assumes your helpers exist in your codebase)
# ------------------------------------------------------------
def newton_solver_scipy_fallback(
    assemble_fn,
    rhs_fn,
    u0: np.ndarray,
    dirichlet_dofs: np.ndarray | None = None,
    dirichlet_vals: np.ndarray | None = None,
    *assemble_args,
    tol: float = 1e-2,
    maxit: int = 50,
    alpha: float = 1.0,
    rhs_args: tuple = (),
    jac_conditioner=False
):
    if dirichlet_dofs is None or len(dirichlet_dofs) == 0:
        return _newton_no_dirichlet_bc(
            assemble_fn, rhs_fn, u0,
            *assemble_args,
            rhs_args=rhs_args,
            tol=tol, maxit=maxit, alpha=alpha, jac_conditioner=jac_conditioner
        )
    else:
        return _newton_with_dirichlet_bc(
            assemble_fn, rhs_fn, u0,
            dirichlet_dofs, dirichlet_vals,
            *assemble_args,
            rhs_args=rhs_args,
            tol=tol, maxit=maxit, alpha=alpha,
            jac_conditioner=jac_conditioner,
        )


def _newton_with_dirichlet_bc(
    assemble_fn, rhs_fn, u0, dofs, vals,
    *assemble_args,
    rhs_args=(),
    tol=1e-2,
    maxit=50,
    jac_conditioner=False,
    alpha=1.0,
):
    from skfem import condense, solve
    # These come from your codebase (leave as-is)
    # - build_pc_amgsa
    # - solver_iter_pcg
    u = u0.copy()
    pc = None
    u[dofs] = vals
    free_dofs = np.setdiff1d(np.arange(len(u)), dofs)
    delta = np.zeros_like(u)

    for itr in range(maxit):
        delta.fill(0.0)

        J = assemble_fn(u, *assemble_args)
        RHS = rhs_fn(u, *rhs_args)

        if jac_conditioner:
            Jc, RHSc, _, free = condense(J, RHS, D=dofs)
            if itr < 2:
                pc = build_pc_amgsa(Jc)
            delta_free = solve(
                Jc, -RHSc,
                solver=solver_iter_pcg(verbose=False, M=pc, tol=1e-16, atol=1e-16)
            )
            delta[free] = delta_free
        else:
            if (itr + 1) % 40 == 0:
                alpha *= 0.5
                print(f"Iteration {itr}: reducing α → {alpha}")

            Jc, RHSc, _, _ = condense(J, -RHS, I=free_dofs)
            delta_free = solve(Jc, RHSc)
            delta[free_dofs] = alpha * delta_free

        u += delta

        if np.linalg.norm(delta) < tol:
            return u

    print("Newton solver did not converge!")
    return u


def _newton_no_dirichlet_bc(
    assemble_fn, rhs_fn, u0,
    *assemble_args,
    rhs_args=(),
    tol=1e-2,
    maxit=50,
    alpha=1.0,
    jac_conditioner=False,
):
    from skfem import solve
    # These come from your codebase (leave as-is)
    # - build_pc_amgsa
    # - solver_iter_pcg
    u = u0.copy()
    pc = None

    for itr in range(maxit):
        J = assemble_fn(u, *assemble_args)
        RHS = rhs_fn(u, *rhs_args)
        u_prev = u.copy()

        if itr < 2 and jac_conditioner:
            pc = build_pc_amgsa(J)

        delta = solve(
            J, -RHS,
            solver=solver_iter_pcg(verbose=False, M=pc, tol=1e-16, atol=1e-16)
        )
        u += delta*alpha

        if np.linalg.norm(u - u_prev) < tol:
            return u

    raise RuntimeError("Newton solver did not converge!")


# ------------------------------------------------------------
# AUTO Newton: PETSc for linear solves if available, else fallback
# ------------------------------------------------------------
def newton_solver(
    assemble_fn,
    rhs_fn,
    u0: np.ndarray,
    dirichlet_dofs: np.ndarray | None = None,
    dirichlet_vals: np.ndarray | None = None,
    *assemble_args,
    rhs_args: tuple = (),
    tol: float = 1e-2,
    maxit: int = 50,
    alpha: float = 1.0,
    jac_conditioner: bool = False,   # used only by SciPy fallback
    # PETSc KSP options (used only if PETSc import works)
    ksp_type: str = "cg",
    pc_type: str = "ilu",
    ksp_rtol: float = 1e-8,
    ksp_atol: float = 0.0,
    ksp_max_it: int = 2000,
    petsc_options: dict | None = None,
    reuse_ksp: bool = True,
    force_backend: str = "auto",  # "auto" | "petsc" | "scipy"
):
    """
    If PETSc is available (and not forced off), uses PETSc KSP for the *linear solves only*.
    Otherwise, falls back to your existing SciPy-based Newton implementation.
    """
    use_petsc = _petsc_is_available() and (force_backend in ("auto", "petsc"))

    if force_backend == "scipy":
        use_petsc = False
    if force_backend == "petsc" and not _petsc_is_available():
        # user forced PETSc but it's not installed -> fallback cleanly
        use_petsc = False

    if not use_petsc:
        # Fallback to your previous implementation (unchanged behavior)
        return newton_solver_scipy_fallback(
            assemble_fn, rhs_fn, u0,
            dirichlet_dofs=dirichlet_dofs,
            dirichlet_vals=dirichlet_vals,
            *assemble_args,
            tol=tol, maxit=maxit, alpha=alpha,
            rhs_args=rhs_args,
            jac_conditioner=jac_conditioner
        )

    # --- PETSc-linear-solve path ---
    from skfem import condense
    import scipy.sparse as sp

    u = np.array(u0, dtype=np.float64, copy=True)

    dofs = None if dirichlet_dofs is None or len(dirichlet_dofs) == 0 else np.asarray(dirichlet_dofs)
    vals = None if dofs is None else np.asarray(dirichlet_vals)

    if dofs is not None:
        u[dofs] = vals
        free_dofs = np.setdiff1d(np.arange(u.size), dofs)
    else:
        free_dofs = None

    ksp_cache = {} if reuse_ksp else None

    for itr in range(maxit):
        J = assemble_fn(u, *assemble_args)
        R = rhs_fn(u, *rhs_args)

        if sp.issparse(J) and not sp.isspmatrix_csr(J):
            J = J.tocsr()

        if dofs is not None:
            # Reduced system for free dofs
            Jc, Rc, _, _ = condense(J, -R, I=free_dofs)
            if sp.issparse(Jc) and not sp.isspmatrix_csr(Jc):
                Jc = Jc.tocsr()

            delta_free = petsc_solve_csr(
                Jc, Rc,
                ksp_type=ksp_type, pc_type=pc_type,
                rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it,
                petsc_options=petsc_options,
                cache=ksp_cache,
            )

            delta = np.zeros_like(u)
            delta[free_dofs] = alpha * delta_free
            delta[dofs] = 0.0
        else:
            b = -R
            delta = petsc_solve_csr(
                J, b,
                ksp_type=ksp_type, pc_type=pc_type,
                rtol=ksp_rtol, atol=ksp_atol, max_it=ksp_max_it,
                petsc_options=petsc_options,
                cache=ksp_cache,
            )
            delta = alpha * delta

        u += delta

        if dofs is not None:
            u[dofs] = vals

        if np.linalg.norm(delta) < tol:
            return u

    raise RuntimeError("Newton did not converge within maxit.")
