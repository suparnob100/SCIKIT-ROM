"""
Utilities for ROM workflows.

The module includes functions for:
- snapshot index splitting for training and testing
- parameter sampling (Sobol, Latin hypercube, Gaussian)
- basis updates by deflation and augmentation
- reconstruction from reduced to full coordinates
- ROM data save/load helpers
- Newton solvers for reduced and hyper-reduced systems
- residual collection routines used in hyperreduction

Notes
-----
Authors: Suparno Bhattacharyya; Ali Hamza Abidi Syed
"""
"""
Utilities used in ROM workflows.

Contents
--------
- Snapshot index splitting for training and testing
- Sample generation (Sobol, LHS, Gaussian)
- Basis update by deflation and augmentation
- Reconstruction from reduced coordinates to full coordinates
- ROM data save/load helpers
- Newton solvers used in ROM and hyper-ROM steps
- Residual collection helpers for hyperreduction
"""
from skrom.utils.imports import *
from scipy.linalg.blas import dgemm, sgemm


def train_test_split(N_snap, N_sel=None, train_percentage=0.8):
    """
    Create boolean masks that split snapshot indices into train and test sets.

    Parameters
    ----------
    N_snap : int
        Count of snapshots.
    N_sel : int or None
        Count of indices to select before the split. If None, use all indices.
    train_percentage : float
        Fraction assigned to the train set.

    Returns
    -------
    train_mask : ndarray[bool], shape (N_snap,)
        Mask for train indices.
    test_mask : ndarray[bool], shape (N_snap,)
        Mask for test indices.
    """
    indices = np.random.permutation(N_snap)

    if N_sel is not None:
        indices = np.random.choice(indices, N_sel, replace=False)

    train_set_size = int(N_snap * train_percentage)

    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    train_mask[indices[:train_set_size]] = True
    test_mask[indices[train_set_size:]] = True

    return train_mask, test_mask


def latin_hypercube_train_test_split(N_snap, train_percentage=0.8):
    """
    Create train/test masks by ordering indices from a Latin hypercube draw.

    Parameters
    ----------
    N_snap : int
        Count of snapshots.
    train_percentage : float
        Fraction assigned to the train set.

    Returns
    -------
    train_mask : ndarray[bool], shape (N_snap,)
        Mask for train indices.
    test_mask : ndarray[bool], shape (N_snap,)
        Mask for test indices.
    """
    LHS_indices = lhs(N_snap, samples=N_snap, criterion="maximin")
    LHS_indices = np.argsort(LHS_indices[:, 0])

    train_set_size = int(N_snap * train_percentage)

    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    train_mask[LHS_indices[:train_set_size]] = True
    test_mask[LHS_indices[train_set_size:]] = True

    return train_mask, test_mask


def sobol_train_test_split(N_snap, train_percentage=0.8):
    """
    Create train/test masks by ordering indices from a Sobol draw.

    Parameters
    ----------
    N_snap : int
        Count of snapshots.
    train_percentage : float
        Fraction assigned to the train set.

    Returns
    -------
    train_mask : ndarray[bool], shape (N_snap,)
        Mask for train indices.
    test_mask : ndarray[bool], shape (N_snap,)
        Mask for test indices.
    """
    m = int(np.ceil(np.log2(N_snap)))
    sobol_gen = Sobol(d=1)

    sobol_indices = sobol_gen.random_base2(m=m)
    sobol_indices = sobol_indices.flatten()[:N_snap]
    sobol_indices = np.argsort(sobol_indices)

    train_set_size = int(N_snap * train_percentage)

    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    train_mask[sobol_indices[:train_set_size]] = True
    test_mask[sobol_indices[train_set_size:]] = True

    return train_mask, test_mask


def generate_sobol(dimensions, num_points, bounds):
    """
    Generate Sobol samples and scale them to bounds.

    Parameters
    ----------
    dimensions : int
        Count of parameters.
    num_points : int
        Count of samples. Input is used as 2**m with m = log2(num_points).
    bounds : list[tuple[float, float]]
        (lower, upper) for each parameter.

    Returns
    -------
    scaled_samples : ndarray, shape (num_points, dimensions)
        Samples in parameter space.
    """
    sobol = Sobol(d=dimensions)
    samples = sobol.random_base2(m=int(np.log2(num_points)))
    scaled_samples = np.empty_like(samples)

    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower

    return scaled_samples


def generate_lhs(dimensions, num_points, bounds):
    """
    Generate Latin hypercube samples and scale them to bounds.

    Parameters
    ----------
    dimensions : int
        Count of parameters.
    num_points : int
        Count of samples.
    bounds : list[tuple[float, float]]
        (lower, upper) for each parameter.

    Returns
    -------
    scaled_samples : ndarray, shape (num_points, dimensions)
        Samples in parameter space.
    """
    lhs_dist = LatinHypercube(d=dimensions)
    samples = lhs_dist.random(n=num_points)
    scaled_samples = np.empty_like(samples)

    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower

    return scaled_samples


def generate_gaussian_samples(dimensions, num_points, bounds):
    """
    Generate samples from normal draws based on bounds.

    Mean is (lower+upper)/2.
    Std is (upper-lower)/5.

    Parameters
    ----------
    dimensions : int
        Count of parameters.
    num_points : int
        Count of samples.
    bounds : list[tuple[float, float]]
        (lower, upper) for each parameter.

    Returns
    -------
    samples : ndarray, shape (num_points, dimensions)
        Samples from normal draws. No clipping is applied.
    """
    samples = np.zeros((num_points, dimensions))
    means = []
    std_devs = []

    for lower, upper in bounds:
        mean = (upper + lower) / 2
        std_dev = (upper - lower) / 5
        means.append(mean)
        std_devs.append(std_dev)

    for i in range(dimensions):
        samples[:, i] = np.random.normal(loc=means[i], scale=std_devs[i], size=num_points)

    return samples


def update_basis(V, W_mu, max_modes=5):
    """
    Update a basis by deflating snapshots and appending modes from an SVD.

    Steps
    -----
    1) Deflate snapshots: W_deflated = W_mu - V(V^T W_mu)
    2) Compute SVD of W_deflated
    3) Append up to max_modes left singular vectors
    4) Re-orthonormalize by QR

    Parameters
    ----------
    V : ndarray, shape (N_h, r_old)
        Basis matrix.
    W_mu : ndarray, shape (N_h, N_t)
        Snapshot matrix for one parameter value.
    max_modes : int
        Cap on appended modes.

    Returns
    -------
    V_new : ndarray, shape (N_h, r_old + k)
        Updated basis matrix.
    """
    W_deflated = W_mu - V @ (V.T @ W_mu)

    U_new, _, _ = np.linalg.svd(W_deflated, full_matrices=False)

    V_combined = np.hstack([V, U_new[:, :max_modes]])

    V_new, _ = qr(V_combined, mode="economic")

    projection_error = np.linalg.norm(W_mu - V_new @ (V_new.T @ W_mu))
    print("Projection error after update (relative):", projection_error / np.linalg.norm(W_mu))

    return V_new


def reconstruct_solution(u_reduced, V_sel, mean):
    """
    Map reduced coordinates to full coordinates and add the mean.

    Parameters
    ----------
    u_reduced : ndarray
        Reduced coordinates. Shape can be (r,) or (n, r) depending on usage.
    V_sel : ndarray, shape (N_h, r)
        Basis matrix used for reconstruction.
    mean : ndarray, shape (N_h,)
        Mean field used in centering.

    Returns
    -------
    u_complete : ndarray
        Reconstructed field with mean added.
    """
    u_full = V_sel @ u_reduced if V_sel.shape[1] == u_reduced.shape[0] else V_sel @ u_reduced.T

    m = mean[None, :] if u_full.ndim == 2 and mean.ndim != u_full.ndim else mean

    u_complete = u_full.T + m

    return u_complete


def rom_data_gen(save_kw, problem_path):
    """
    Save ROM outputs to a ROM_data folder.

    This function stores:
    - fos_solutions.npy
    - ROM_simulation_data.npz (all other items from save_kw)

    Parameters
    ----------
    save_kw : dict
        Data to save. Must contain key "fos_solutions".
    problem_path : str or Path
        Path to the problem folder.

    Raises
    ------
    KeyError
        If "fos_solutions" is not present in save_kw.
    """
    rom_dir = Path(problem_path) / "ROM_data"
    rom_dir.mkdir(parents=True, exist_ok=True)

    try:
        sol = save_kw.pop("fos_solutions")
    except KeyError:
        raise KeyError("rom_data_gen requires 'fos_solutions' in save_kw")

    fos_path = rom_dir / "fos_solutions.npy"
    np.save(fos_path, np.array(sol, copy=False))
    print(f"Saved full-order solution → {fos_path.name}")

    npz_path = rom_dir / "ROM_simulation_data.npz"
    np.savez_compressed(npz_path, **save_kw)
    print(f"Saved ROM data             → {npz_path.name}")


def load_rom_data(self, rom_data_dir: str | Path | None = None):
    """
    Load ROM data from a ROM_data folder.

    Behavior
    --------
    - If self is None, return (fos_solutions, sim_data_dict).
    - If self is not None, set attributes on self and return None.

    Parameters
    ----------
    self : object or None
        Target instance that holds problem_name, or None for return mode.
    rom_data_dir : str | Path | None
        Path to ROM_data, or a module path string, or None for auto path.

    Returns
    -------
    (fos_solutions, sim_data) : tuple
        Returned only when self is None.
    """
    if self is None:
        if rom_data_dir is None:
            base = Path(__file__).resolve().parent
            rom_dir = base / self.problem_name / "ROM_data"
        else:
            rom_dir = (
                Path(rom_data_dir)
                if not isinstance(rom_data_dir, str) or Path(rom_data_dir).exists()
                else Path(importlib.import_module(str(rom_data_dir)).__file__).parent
            )

        fos_solutions = np.load(rom_dir / "fos_solutions.npy", allow_pickle=True)
        data = np.load(rom_dir / "ROM_simulation_data.npz", allow_pickle=True)
        sim_data = {name: val for name, val in data.items()}
        print(f"[load_rom_data] loaded from {rom_dir}")
        return fos_solutions, sim_data
    else:
        if rom_data_dir is None:
            base = Path(__file__).resolve().parent
            rom_dir = base / self.problem_name / "ROM_data"
        else:
            rom_dir = (
                Path(rom_data_dir)
                if not isinstance(rom_data_dir, str) or Path(rom_data_dir).exists()
                else Path(importlib.import_module(str(rom_data_dir)).__file__).parent
            )

        self.fos_solutions = np.load(rom_dir / "fos_solutions.npy", allow_pickle=True)
        data = np.load(rom_dir / "ROM_simulation_data.npz", allow_pickle=True)
        for name, val in data.items():
            setattr(self, name, val)
        print(f"[load_rom_data] loaded from {rom_dir}")


def _ensure_csr(mat):
    """
    Convert matrix input to CSR sparse matrix.

    Parameters
    ----------
    mat : array_like or sparse matrix or 0-d object array
        Matrix input.

    Returns
    -------
    csr : scipy.sparse.csr_matrix
        CSR matrix view or copy.
    """
    if isinstance(mat, np.ndarray) and mat.dtype == object and mat.shape == ():
        mat = mat.item()
    return mat.tocsr() if issparse(mat) else csr_matrix(np.asarray(mat, float))


def newton_hyper_rom_solver(assemble_func, u, tol=3e-2, maxit=200, param=None):
    """
    Newton solve for a reduced system defined by assemble_func.

    assemble_func(u, param) must return (A, y) such that:
        A * delta = -y

    Parameters
    ----------
    assemble_func : callable
        Function with signature (A, y) = assemble_func(u, param).
    u : ndarray
        Initial iterate. Updated in place.
    tol : float
        Stop threshold on ||delta||.
    maxit : int
        Iteration cap.
    param : any
        Input passed to assemble_func.

    Returns
    -------
    u : ndarray
        Final iterate.
    """
    alpha = 1.0
    damp_freq = 40

    for itr in range(maxit):
        A, y = assemble_func(u, param)

        if (itr > 0 and itr % damp_freq == 0) or (itr > 3 and step_norm > 1e4):
            alpha *= 0.5
            print(f"[ROM Newton] iter {itr}: reducing α → {alpha:.2e}")

        delta = np.linalg.solve(A, -y)

        step_norm = np.linalg.norm(delta)

        u += alpha * delta

        print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e}")

        if step_norm < tol:
            return u
        elif itr == maxit - 1:
            print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e} (not converged)")
            return u


# def newton_hyper_rom_solver2(
#     J_rom_fn,
#     rhs_rom_fn,
#     u0: np.ndarray,
#     *assemble_args,
#     tol: float = 1e-2,
#     maxit: int = 50,
#     alpha: float = 1.0,
#     rhs_args: tuple = (),
# ):
#     """
#     Newton solve for a reduced system given separate Jacobian and residual calls.

#     The iteration solves:
#         J(u) * delta = -R(u)

#     Parameters
#     ----------
#     J_rom_fn : callable
#         Function that returns J(u).
#     rhs_rom_fn : callable
#         Function that returns R(u).
#     u0 : ndarray
#         Initial iterate.
#     *assemble_args : tuple
#         Positional inputs forwarded to J_rom_fn.
#     tol : float
#         Stop threshold on ||u - u_prev||.
#     maxit : int
#         Iteration cap.
#     alpha : float
#         Step scaling.
#     rhs_args : tuple
#         Positional inputs forwarded to rhs_rom_fn.

#     Returns
#     -------
#     u : ndarray
#         Final iterate.
#     """
#     u = u0.copy()
#     damp_int = 40

#     for itr in range(maxit):
#         J_rom = J_rom_fn(u, *assemble_args)
#         RHS_rom = rhs_rom_fn(u, *rhs_args)
#         u_prev = u.copy()

#         if itr > 0 and itr % damp_int == 0:
#             alpha *= 0.5
#             print(f"[ROM Newton] iter {itr}: reducing α → {alpha:.2e}")

#         delta = np.linalg.solve(J_rom, -RHS_rom)
#         u += alpha * delta

#         if np.linalg.norm(u - u_prev) < tol:
#             return u
#         elif itr == maxit - 1:
#             print(f"[Newton] Iter {itr:2d} (not converged)")
#             return u


# def newton_solver_rom(
#     assemble_func,
#     u_rom,
#     *args,
#     alpha: float = 1.0,
#     tol: float = 1e-3,
#     maxit: int = 100,
#     use_lu: bool = False,
#     jac_tol: float = 1e-1,
#     **kwargs,
# ):
#     """
#     Newton solve for a reduced system with two modes.

#     Mode A (use_lu=True)
#     --------------------
#     - Assemble J and R in reduced coordinates
#     - Reuse LU factors until ||J - J_prev||/||J_prev|| exceeds jac_tol
#     - Stop when ||delta|| < tol
#     - Raise RuntimeError on non-convergence

#     Mode B (use_lu=False)
#     ---------------------
#     - Assemble (A, y) and solve A * delta = -y each step
#     - Apply step scaling alpha, with alpha halving every 40 steps
#     - Return on ||delta|| < tol, else return last iterate at maxit

#     Parameters
#     ----------
#     assemble_func : callable
#         If use_lu=True: (J_rom, RHS_rom) = assemble_func(u_rom, *args, **kwargs)
#         If use_lu=False: (A, y) = assemble_func(u_rom, *args, **kwargs)
#     u_rom : array_like
#         Initial iterate in reduced coordinates.
#     *args : tuple
#         Inputs forwarded to assemble_func.
#     alpha : float
#         Step scaling for mode B.
#     tol : float
#         Stop threshold on ||delta|| (mode A) or ||delta|| (mode B via step_norm).
#     maxit : int
#         Iteration cap.
#     use_lu : bool
#         Select mode A when True, else mode B.
#     jac_tol : float
#         Threshold for Jacobian refactor in mode A.
#     **kwargs : dict
#         Keyword inputs forwarded to assemble_func.

#     Returns
#     -------
#     u_rom : array_like
#         Final iterate.
#     """
#     if use_lu:
#         prev_J = None
#         lu_factors = None

#         for itr in range(maxit):
#             J_rom, RHS_rom = assemble_func(u_rom, *args, **kwargs)

#             if prev_J is None:
#                 lu_factors = lu_factor(J_rom)
#                 prev_J = J_rom.copy()
#             else:
#                 rel_change = (
#                     np.linalg.norm(J_rom - prev_J, ord="fro")
#                     / np.linalg.norm(prev_J, ord="fro")
#                 )
#                 if rel_change > jac_tol:
#                     lu_factors = lu_factor(J_rom)
#                     prev_J = J_rom.copy()

#             delta = lu_solve(lu_factors, -RHS_rom)
#             u_rom += delta

#             if np.linalg.norm(delta) < tol:
#                 return u_rom

#         raise RuntimeError(f"Newton (LU) did not converge in {maxit} iterations")

#     else:
#         damp_freq = 40

#         for itr in range(maxit):
#             A, y = assemble_func(u_rom, *args, **kwargs)

#             if itr > 0 and itr % damp_freq == 0:
#                 alpha *= 0.5
#                 print(f"[ROM Newton] iter {itr}: reducing α → {alpha:.2e}")

#             delta = np.linalg.solve(A, -y)
#             step_norm = np.linalg.norm(delta)

#             u_rom += alpha * delta

#             print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e}")

#             if step_norm < tol:
#                 return u_rom
#             elif itr == maxit - 1:
#                 print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e} (not converged)")
#                 return u_rom




def _petsc_available() -> bool:
    try:
        from petsc4py import PETSc  # noqa: F401
        return True
    except Exception:
        return False


def _iter_solve_scipy(
    A: np.ndarray,
    b: np.ndarray,
    *,
    method: str = "cg",          # "cg" | "gmres" | "minres"
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxit: int = 500,
    M=None,                      # optional preconditioner (scipy LinearOperator)
):
    from scipy.sparse.linalg import cg, gmres, minres

    method = method.lower()
    if method == "cg":
        x, info = cg(A, b, rtol=rtol, atol=atol, maxiter=maxit, M=M)
    elif method == "gmres":
        x, info = gmres(A, b, rtol=rtol, atol=atol, maxiter=maxit, M=M)
    elif method == "minres":
        x, info = minres(A, b, rtol=rtol, maxiter=maxit, M=M)
    else:
        raise ValueError('method must be "cg", "gmres", or "minres"')

    if info != 0:
        raise RuntimeError(f"SciPy {method} did not converge (info={info})")
    return np.asarray(x, dtype=np.float64)


def _iter_solve_petsc(
    A: np.ndarray,
    b: np.ndarray,
    *,
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxit: int = 500,
    petsc_options: dict | None = None,
):
    if not _petsc_available():
        raise RuntimeError("PETSc not available")

    from petsc4py import PETSc

    A = np.asarray(A)
    b = np.asarray(b)

    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square, got {A.shape}")
    if b.shape != (n,):
        raise ValueError(f"b must have shape ({n},), got {b.shape}")

    # Dense PETSc Mat (ROM is typically dense)
    mat = PETSc.Mat().createDense([n, n], array=A, comm=PETSc.COMM_SELF)
    mat.assemble()

    rhs = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    sol = PETSc.Vec().createSeq(n, comm=PETSc.COMM_SELF)
    rhs.setArray(b)

    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)

    ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxit)

    if petsc_options:
        opts = PETSc.Options()
        for k, v in petsc_options.items():
            if v is None:
                opts.setValue(k, None)
            else:
                opts.setValue(k, str(v))

    ksp.setFromOptions()
    ksp.solve(rhs, sol)

    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(f"PETSc KSP diverged (reason={reason}, its={ksp.getIterationNumber()})")

    return sol.getArray().copy()


def solve_linear(
    A,
    b,
    *,
    backend: str = "numpy",      # "numpy" | "scipy_iter" | "petsc"
    method: str = "cg",          # for iterative backends
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxit: int = 500,
    petsc_pc: str = "jacobi",
    petsc_options: dict | None = None,
):
    """
    Solve A x = b with a chosen backend.

    Notes:
      - CG requires SPD.
      - MINRES requires symmetric (can be indefinite).
      - GMRES works for general matrices.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    backend = backend.lower()
    if backend == "numpy":
        return np.linalg.solve(A, b)

    if backend == "scipy_iter":
        return _iter_solve_scipy(A, b, method=method, rtol=rtol, atol=atol, maxit=maxit)

    if backend == "petsc":
        return _iter_solve_petsc(
            A, b,
            ksp_type=method,
            pc_type=petsc_pc,
            rtol=rtol, atol=atol, maxit=maxit,
            petsc_options=petsc_options,
        )

    raise ValueError('backend must be "numpy", "scipy_iter", or "petsc"')


def newton_solver_rom(
    assemble_func,
    u0,
    *args,
    tol: float = 1e-2,
    maxit: int = 100,
    alpha: float = 1.0,
    damp_freq: int = 40,
    linear_backend: str = "numpy",    # "numpy" | "scipy_iter" | "petsc"
    linear_method: str = "cg",        # cg/gmres/minres (SciPy) or cg/gmres/minres (PETSc)
    linear_rtol: float = 1e-8,
    linear_atol: float = 0.0,
    linear_maxit: int = 500,
    petsc_pc: str = "jacobi",
    petsc_options: dict | None = None,
    verbose: bool = True,
    **kwargs,
):
    """
    Single-mode ROM Newton: always solve (A * delta = -y) each iteration
    using the selected linear backend (dense direct or iterative).
    """
    u = np.array(u0, dtype=np.float64, copy=True)
    cur_alpha = float(alpha)

    for itr in range(maxit):
        A, y = assemble_func(u, *args, **kwargs)
        A = np.asarray(A)
        y = np.asarray(y)

        if itr > 0 and damp_freq and (itr % damp_freq == 0):
            cur_alpha *= 0.5
            if verbose:
                print(f"[ROM Newton] iter {itr}: reducing α → {cur_alpha:.2e}")

        delta = solve_linear(
            A, -y,
            backend=linear_backend,
            method=linear_method,
            rtol=linear_rtol,
            atol=linear_atol,
            maxit=linear_maxit,
            petsc_pc=petsc_pc,
            petsc_options=petsc_options,
        )

        step = np.linalg.norm(delta)
        u += cur_alpha * delta

        if verbose and (itr % 10 == 0 or step < tol):
            print(f"[ROM Newton] Iter {itr:2d}, step norm = {step:.3e}")

        if step < tol:
            return u

    if verbose:
        print(f"[ROM Newton] reached maxit={maxit} (not converged)")
    return u


################
# Hyperreduction
################

import numpy as np


def newton_hyper_rom_solver2(
    J_rom_fn,
    rhs_rom_fn,
    u0: np.ndarray,
    *J_args,
    tol: float = 1e-2,
    maxit: int = 50,
    alpha: float = 1.0,
    damp_freq: int = 40,
    rhs_args: tuple = (),
    # linear solve controls (same as newton_solver_rom)
    linear_backend: str = "numpy",   # "numpy" | "scipy_iter" | "petsc"
    linear_method: str = "cg",       # "cg" (SPD) | "gmres" | "minres"
    linear_rtol: float = 1e-8,
    linear_atol: float = 0.0,
    linear_maxit: int = 500,
    petsc_pc: str = "ilu",
    petsc_options: dict | None = None,
    verbose: bool = True,
):
    """
    Hyper-ROM Newton wrapper:
      J_rom_fn(u, *J_args) -> J(u)   (dense reduced Jacobian)
      rhs_rom_fn(u, *rhs_args) -> R(u) (reduced residual)
    Solves: J(u) * delta = -R(u)
    """

    def assemble_func(u, *_, **__):
        J = J_rom_fn(u, *J_args)
        R = rhs_rom_fn(u, *rhs_args)
        return J, R

    return newton_solver_rom(
        assemble_func,
        u0,
        tol=tol,
        maxit=maxit,
        alpha=alpha,
        damp_freq=damp_freq,
        linear_backend=linear_backend,
        linear_method=linear_method,
        linear_rtol=linear_rtol,
        linear_atol=linear_atol,
        linear_maxit=linear_maxit,
        petsc_pc=petsc_pc,
        petsc_options=petsc_options,
        verbose=verbose,
    )


def collect_residuals(
    NLS_train_ms,
    NLS_train_mean,
    V_sel,
    reconstruct_solution,
    Residual,
    training_params,
    assemble_kwargs,
    extra_kwargs=None,
    hyper_basis=None,
):
    """
    Evaluate and store residual samples used in hyperreduction training.

    For each snapshot:
    - project snapshot to reduced coordinates
    - reconstruct a field
    - build keyword arguments by assemble_kwargs
    - call Residual.hyperreduction(**kw)
    - stack results along axis 0

    Parameters
    ----------
    NLS_train_ms : ndarray
        Mean-shifted snapshots.
    NLS_train_mean : ndarray
        Mean field.
    V_sel : ndarray
        Basis used for projection and reconstruction.
    reconstruct_solution : callable
        Function that maps reduced coordinates to full coordinates.
    Residual : object
        Object with method hyperreduction(**kw) that returns an array.
    training_params : sequence
        Parameter list aligned with NLS_train_ms.
    assemble_kwargs : callable
        Function that maps (u_recon, param) to a dict for Residual.hyperreduction.
    extra_kwargs : dict or None
        Dict merged into kw.
    hyper_basis : object or None
        If given, call hyper_basis.interpolate(u_recon) before assemble_kwargs.

    Returns
    -------
    q_mus : ndarray
        Stacked residual samples.
    """
    q_mus = None
    step = 1

    for i, u_arr_ms in enumerate(NLS_train_ms):
        u_red = V_sel.T @ u_arr_ms
        u_recon = reconstruct_solution(u_red, V_sel, NLS_train_mean)

        if hyper_basis is not None:
            u_rec = hyper_basis.interpolate(u_recon.flatten())
        else:
            u_rec = u_recon.flatten()

        kw = assemble_kwargs(u_rec, training_params[i])
        kw.update(extra_kwargs or {})

        q = Residual.hyperreduction(**kw).T.copy()

        if q_mus is None:
            q_mus = q
        else:
            q_mus = np.concatenate((q_mus, q), axis=0)

    return q_mus


def collect_residuals_t(
    NLS_train_ms,
    NLS_train_mean,
    V_sel,
    reconstruct_solution,
    Residual,
    training_params,
    assemble_kwargs,
    snapshot_downsampling=1,
    extra_kwargs=None,
    hyper_basis=None,
):
    """
    Evaluate residual samples for a set of snapshot groups.

    This variant loops over an outer list/array of snapshot groups and
    applies downsampling by index.

    Parameters
    ----------
    NLS_train_ms : sequence
        Collection of snapshot groups.
    NLS_train_mean : sequence
        Collection of mean fields aligned with snapshot groups.
    V_sel : ndarray
        Basis used for projection and reconstruction.
    reconstruct_solution : callable
        Function that maps reduced coordinates to full coordinates.
    Residual : object
        Object with method hyperreduction(**kw) that returns an array.
    training_params : sequence
        Parameter list. Kept for interface consistency.
    assemble_kwargs : callable
        Function that maps a reconstructed field to a dict for Residual.hyperreduction.
    snapshot_downsampling : int
        Keep one snapshot per this interval.
    extra_kwargs : dict or None
        Dict merged into kw.
    hyper_basis : object or None
        If given, call hyper_basis.interpolate(u_recon) before assemble_kwargs.

    Returns
    -------
    q_mus : ndarray
        Stacked residual samples.
    """
    q_mus = None

    for k, NLS_train_ms_p in enumerate(NLS_train_ms):
        for i, u_arr_ms in enumerate(NLS_train_ms_p):
            np.random.shuffle(NLS_train_ms_p)

            if i % snapshot_downsampling == 0:
                u_red = V_sel.T @ u_arr_ms
                u_recon = reconstruct_solution(u_red, V_sel, NLS_train_mean[k])

                if hyper_basis is not None:
                    u_rec = hyper_basis.interpolate(u_recon.flatten())
                else:
                    u_rec = u_recon.flatten()

                kw = assemble_kwargs(u_rec)
                kw.update(extra_kwargs or {})

                q = Residual.hyperreduction(**kw).T.copy()

                if q_mus is None:
                    q_mus = q
                else:
                    q_mus = np.concatenate((q_mus, q), axis=0)

    return q_mus


def select_elements_and_gauss_weights(n_gauss_points, element_indices, weights):
    """
    Build a mapping from element index to a weight list per Gauss point.

    Input indices are Gauss-point indices. The mapping includes one entry per
    element that appears in element_indices. For each mapped element, the list
    length equals n_gauss_points. Unselected Gauss points keep weight 0.0.

    Parameters
    ----------
    n_gauss_points : int
        Gauss point count per element.
    element_indices : sequence[int]
        Gauss-point indices selected by a sampling step.
    weights : sequence[float]
        Weights aligned with element_indices.

    Returns
    -------
    element_to_gauss_weights : dict[int, list[float]]
        Map element_id -> list of weights with length n_gauss_points.
    """
    element_to_gauss_weights = {}

    for idx, weight in zip(element_indices, weights):
        element_idx = idx // n_gauss_points
        gauss_point_idx = idx % n_gauss_points

        if element_idx not in element_to_gauss_weights:
            element_to_gauss_weights[element_idx] = [0.0] * n_gauss_points

        element_to_gauss_weights[element_idx][gauss_point_idx] = weight

    return element_to_gauss_weights


def compute_nonlinear_snapshots(
    non_linear_func,
    fos_solutions,
    param_list,
) -> np.ndarray:
    """
    Evaluate a function on each (solution, parameter) pair and stack results.

    Parameters
    ----------
    non_linear_func : callable
        Function with signature non_linear_func(u=<solution>, param=<param>).
    fos_solutions : sequence
        Sequence of full-order solutions.
    param_list : sequence
        Sequence of parameters aligned with fos_solutions.

    Returns
    -------
    F_nl : ndarray
        Stacked outputs of non_linear_func.
    """
    F_nl = []
    for u, param in zip(fos_solutions, param_list):
        rhs_snapshot = non_linear_func(u=u, param=param)
        F_nl.append(np.copy(rhs_snapshot))
    return np.asarray(F_nl)
