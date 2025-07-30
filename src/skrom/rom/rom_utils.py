"""
General-purpose utilities for snapshot splitting, sampling, basis updates, data I/O,
and Newton solvers in reduced‐order modeling (ROM) workflows.

This module provides:

  - Snapshot train/test splitting routines:
      * `train_test_split`, `latin_hypercube_train_test_split`, `sobol_train_test_split`
  - Sample generators:
      * `generate_sobol`, `generate_lhs`, `generate_gaussian_samples`
  - Basis management:
      * `update_basis` – deflation and augmentation of reduced bases
  - Solution reconstruction:
      * `reconstruct_solution` – expand reduced vectors back to full order
  - ROM data persistence:
      * `rom_data_gen`, `load_rom_data`
  - Newton solvers for ROM systems:
      * `newton_hyper_rom_solver`, `newton_solver_rom`

Together, these functions support data preparation, sampling design,
basis enrichment, I/O, and nonlinear solves in full‐to‐reduced‐order pipelines.
"""

from skrom.utils.imports import *

def train_test_split(N_snap, N_sel=None, train_percentage=0.8):
    """
    Split snapshot indices into training and testing masks.

    Parameters
    ----------
    N_snap : int
        Total number of snapshots.
    N_sel : int or None, optional
        Number of snapshots to select before splitting. If None, all snapshots are used. Default is None.
    train_percentage : float, optional
        Proportion of snapshots to include in the training set. Defaults to 0.8.

    Returns
    -------
    train_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating training snapshots.
    test_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating testing snapshots.
    """
    # Generate a random permutation of indices from 0 to data_size - 1
    indices = np.random.permutation(N_snap)

    if N_sel is not None:
        indices = np.random.choice(indices, N_sel, replace=False)

    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set the first train_set_size indices to True for the training mask
    train_mask[indices[:train_set_size]] = True

    # Set the remaining indices to True for the testing mask
    test_mask[indices[train_set_size:]] = True

    return train_mask, test_mask


def latin_hypercube_train_test_split(N_snap, train_percentage=0.8):
    """
    Split snapshots into training and testing masks via Latin Hypercube Sampling.

    Parameters
    ----------
    N_snap : int
        Total number of snapshots.
    train_percentage : float, optional
        Proportion of snapshots to include in the training set. Defaults to 0.8.

    Returns
    -------
    train_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating training snapshots.
    test_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating testing snapshots.
    """
    # Generate Latin Hypercube Sampling indices
    LHS_indices = lhs(N_snap, samples=N_snap, criterion='maximin')
    LHS_indices = np.argsort(LHS_indices[:, 0])  # Convert to indices

    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set masks according to LHS indices
    train_mask[LHS_indices[:train_set_size]] = True
    test_mask[LHS_indices[train_set_size:]] = True

    return train_mask, test_mask


def sobol_train_test_split(N_snap, train_percentage=0.8):
    """
    Split snapshots into training and testing masks via Sobol sequence ordering.

    Parameters
    ----------
    N_snap : int
        Total number of snapshots.
    train_percentage : float, optional
        Proportion of snapshots to include in the training set. Defaults to 0.8.

    Returns
    -------
    train_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating training snapshots.
    test_mask : ndarray of bool, shape (N_snap,)
        Boolean mask indicating testing snapshots.
    """
    # Calculate the nearest power of two
    m = int(np.ceil(np.log2(N_snap)))
    sobol_gen = Sobol(d=1)

    # Generate more points if needed and trim to N_snap
    sobol_indices = sobol_gen.random_base2(m=m)
    sobol_indices = sobol_indices.flatten()[:N_snap]  # Trim if longer than N_snap
    sobol_indices = np.argsort(sobol_indices)         # Convert to sorted indices

    # Calculate the number of samples in the training set
    train_set_size = int(N_snap * train_percentage)

    # Initialize boolean masks
    train_mask = np.zeros(N_snap, dtype=bool)
    test_mask = np.zeros(N_snap, dtype=bool)

    # Set masks according to Sobol indices
    train_mask[sobol_indices[:train_set_size]] = True
    test_mask[sobol_indices[train_set_size:]] = True

    return train_mask, test_mask


def generate_sobol(dimensions, num_points, bounds):
    """
    Generate a Sobol sequence scaled to given bounds.

    Parameters
    ----------
    dimensions : int
        Number of dimensions in the Sobol sequence.
    num_points : int
        Number of points in the sequence (must be a power of two).
    bounds : list of tuple of float
        List of (lower, upper) bounds for each dimension.

    Returns
    -------
    scaled_samples : ndarray, shape (num_points, dimensions)
        Sobol sequence samples scaled to the provided bounds.
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
    Generate a Latin Hypercube Sample (LHS) scaled to given bounds.

    Parameters
    ----------
    dimensions : int
        Number of dimensions in the sample.
    num_points : int
        Number of points to generate.
    bounds : list of tuple of float
        List of (lower, upper) bounds for each dimension.

    Returns
    -------
    scaled_samples : ndarray, shape (num_points, dimensions)
        LHS samples scaled to the provided bounds.
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
    Generate Gaussian-distributed samples based on bounds-derived statistics.

    Parameters
    ----------
    dimensions : int
        Number of dimensions.
    num_points : int
        Number of points to generate.
    bounds : list of tuple of float
        List of (lower, upper) bounds for each dimension; means and stds are derived from these.

    Returns
    -------
    samples : ndarray, shape (num_points, dimensions)
        Gaussian-distributed samples without clipping to the original bounds.
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
    Update a reduced basis by appending new modes from deflated snapshots.

    Parameters
    ----------
    V : ndarray, shape (N_h, r_old)
        Current orthonormal reduced basis.
    W_mu : ndarray, shape (N_h, N_t)
        New high-fidelity snapshots for parameter μ.
    max_modes : int, optional
        Maximum number of new modes to append from deflation. Defaults to 5.

    Returns
    -------
    V_new : ndarray, shape (N_h, r_old + k)
        Re-orthonormalized basis combining old and newly added modes.
    """
    # Remove projection onto current basis V (deflation)
    W_deflated = W_mu - V @ (V.T @ W_mu)

    # SVD of deflated snapshots
    U_new, _, _ = np.linalg.svd(W_deflated, full_matrices=False)

    # Combine old and new vectors
    V_combined = np.hstack([V, U_new[:, :max_modes]])

    # QR re-orthonormalization
    V_new, _ = qr(V_combined, mode='economic')

    projection_error = np.linalg.norm(W_mu - V_new @ (V_new.T @ W_mu))
    print("Projection error after update (relative):", projection_error / np.linalg.norm(W_mu))

    return V_new


def reconstruct_solution(u_reduced, V_sel, mean):
    """
    Reconstruct a full-order solution from a reduced solution vector.

    Parameters
    ----------
    u_reduced : ndarray, shape (r,)
        Reduced solution vector.
    V_sel : ndarray, shape (N_h, r)
        Basis matrix for free degrees of freedom.
    mean : ndarray, shape (N_h,)
        Mean vector that was subtracted during snapshot centering.

    Returns
    -------
    u_full : ndarray, shape (N_h,)
        Full-order solution vector, including mean shift.
    """
    # Compute the free part of the solution from the reduced solution.
    u_full = V_sel @ u_reduced + mean
    return u_full


def rom_data_gen(save_kw, problem_path):
    """
    Save ROM simulation data to disk.

    Parameters
    ----------
    save_kw : dict
        Dictionary containing simulation outputs; must include 'fos_solutions'.
    problem_path : str or Path
        Filesystem path to the problem directory.

    Raises
    ------
    KeyError
        If 'fos_solutions' key is missing in save_kw.
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
    Load ROM data from a ROM_data directory or module path.

    Parameters
    ----------
    self : object or None
        If an instance is provided, data is loaded into attributes; if None, data is returned.
    rom_data_dir : str, Path, or None, optional
        Directory or module path to load ROM_data from. Default is None (auto-detect).

    Returns
    -------
    fos_solutions : ndarray
        Loaded full-order solution snapshots.
    sim_data : dict
        Dictionary of loaded simulation data when self is None; otherwise sets attributes on self.
    """
    if self is None:
        if rom_data_dir is None:
            base = Path(__file__).resolve().parent
            rom_dir = base / self.problem_name / "ROM_data"
        else:
            rom_dir = Path(rom_data_dir) if not isinstance(rom_data_dir, str) or Path(rom_data_dir).exists() else Path(importlib.import_module(str(rom_data_dir)).__file__).parent

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
            rom_dir = Path(rom_data_dir) if not isinstance(rom_data_dir, str) or Path(rom_data_dir).exists() else Path(importlib.import_module(str(rom_data_dir)).__file__).parent

        self.fos_solutions = np.load(rom_dir / "fos_solutions.npy", allow_pickle=True)
        data = np.load(rom_dir / "ROM_simulation_data.npz", allow_pickle=True)
        for name, val in data.items():
            setattr(self, name, val)
        print(f"[load_rom_data] loaded from {rom_dir}")

# helper to ensure CSR format
def _ensure_csr(mat):
    if isinstance(mat, np.ndarray) and mat.dtype == object and mat.shape == ():
        mat = mat.item()
    return mat.tocsr() if issparse(mat) else csr_matrix(np.asarray(mat, float))

def newton_hyper_rom_solver(assemble_func, u, tol=3e-2, maxit=200, param=None):
    """
    Solve a hyper-reduced ROM system via Newton's method.

    Parameters
    ----------
    instance : object
        Object with method assemble_hyper_rom_system(u, params) returning (A, y).
    u : ndarray
        Initial reduced solution vector, updated in place.
    tol : float, optional
        Convergence tolerance on the norm of the update. Defaults to 1e-2.
    maxit : int, optional
        Maximum number of Newton iterations. Defaults to 50.
    params : any, optional
        Additional parameters passed to assemble_hyper_rom_system.

    Returns
    -------
    u : ndarray
        Converged reduced solution.

    Raises
    ------
    RuntimeError
        If convergence is not achieved within maxit iterations.
    """
    # for itr in range(maxit):
    #     A, y = assemble_func(u, param)
    #     u_prev_old_red = u.copy()
    #     u += np.linalg.solve(A, -y)
    #     diff = np.linalg.norm(u - u_prev_old_red)
    #     print(f"Iteration {itr}: Residual norm = {diff:.3e}")
    #     if diff < tol:
    #         return u
    # raise RuntimeError("Newton solver did not converge!")


    alpha = 1.0
    damp_freq = 40  # frequency to reduce alpha

    for itr in range(maxit):
        # Reconstruct full-order state via helper

        A, y = assemble_func(u, param)

        if (itr > 0 and itr % damp_freq == 0) or (itr > 3 and step_norm > 1e4):
            alpha *= 0.5
            print(f"[ROM Newton] iter {itr}: reducing α → {alpha:.2e}")

        delta = np.linalg.solve(A, -y)

        step_norm = np.linalg.norm(delta)

        u += alpha*delta

        print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e}")

        if step_norm < tol:
            return u
        
        elif itr == maxit-1:
            print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e} (not converged)")            
            return u

            
    # raise RuntimeError(f"Newton (direct) did not converge in {maxit} iterations")



def newton_solver_rom(
    assemble_func,
    u_rom,
    *args,
    alpha: float = 1.0,
    tol: float = 1e-3,
    maxit: int = 100,
    use_lu: bool = False,
    jac_tol: float = 1e-1,
    **kwargs
):
    """
    Solve a nonlinear reduced-order system via Newton’s method.

    If use_lu=True, uses LU refactorization on the reduced Jacobian.
    If use_lu=False, reconstructs full state and solves directly each iteration.

    Returns:
      - (u_rom,) when use_lu=True
      - (u_full, mean) when use_lu=False
    """
    if use_lu:
        prev_J = None
        lu_factors = None

        for itr in range(maxit):
            J_rom, RHS_rom = assemble_func(u_rom, *args, **kwargs)

            if prev_J is None:
                lu_factors = lu_factor(J_rom)
                prev_J = J_rom.copy()
            else:
                rel_change = (
                    np.linalg.norm(J_rom - prev_J, ord='fro')
                    / np.linalg.norm(prev_J, ord='fro')
                )
                if rel_change > jac_tol:
                    lu_factors = lu_factor(J_rom)
                    prev_J = J_rom.copy()

            delta = lu_solve(lu_factors, -RHS_rom)
            u_rom += delta
            
            if np.linalg.norm(delta) < tol:
                return u_rom

        raise RuntimeError(f"Newton (LU) did not converge in {maxit} iterations")

    else:

        damp_freq = 40  # frequency to reduce alpha

        for itr in range(maxit):
            # Reconstruct full-order state via helper
            A, y = assemble_func(u_rom, *args, **kwargs)

            if itr > 0 and itr % damp_freq == 0:
                alpha *= 0.5
                print(f"[ROM Newton] iter {itr}: reducing α → {alpha:.2e}")

            delta = np.linalg.solve(A, -y)

            step_norm = np.linalg.norm(delta)

            u_rom += alpha*delta

            print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e}")

            if step_norm < tol:
                return u_rom
            
            elif itr == maxit-1:
                print(f"[Newton] Iter {itr:2d}, step norm = {step_norm:.3e} (not converged)")
                return u_rom
            


################
# Hyperreduction
################


def collect_residuals(
    NLS_train_ms,
    NLS_train_mean,
    V_sel,
    reconstruct_solution,
    Residual,
    training_params,
    assemble_kwargs,
    extra_kwargs = None    
):
    """
    Collect reduced evaluations of the ROM residual functional for hyperreduction.
    
    This function processes training snapshots to collect residual evaluations that will
    be used for hyperreduction (reducing computational cost of nonlinear ROM terms).
    
    Parameters
    ----------
    fos_solutions : ndarray
        Full order solutions (not directly used but maintained for interface consistency)
    NLS_train_ms : ndarray, shape (n_snapshots, n_dofs)
        Mean-subtracted training snapshots (temperature fluctuations from mean)
    NLS_train_mean : ndarray, shape (n_dofs,)
        Mean temperature field across all training snapshots
    V_sel : ndarray, shape (n_dofs, n_modes)
        Selected POD basis matrix (reduced basis vectors)
    reconstruct_solution : callable
        Function to reconstruct full-order solution from ROM coefficients
        Signature: u_full = reconstruct_solution(u_reduced, V_sel, u_mean)
    Residual : LinearFormROM object
        ROM residual operator with hyperreduction capability
        Must have method: hyperreduction(prev=solution, k_param=k, q_param=q)
    
    Returns
    -------
    q_mus : ndarray, shape (n_snapshots, n_residual_components)
        Collected residual evaluations for all training snapshots
        Each row contains the hyperreduced residual evaluation for one parameter case
    """
    
    # Initialize storage for residual evaluations
    q_mus = None
    
    # Down-sampling factor for processing snapshots (currently set to 1 = no downsampling)
    step = 1
    
    # Loop through all mean-centered training snapshots
    for i, u_arr_ms in enumerate(NLS_train_ms):
                
        # Project mean-centered snapshot to reduced coordinates
        # This gives the ROM coefficients for this parameter case
        u_red = V_sel.T @ u_arr_ms

        # Reconstruct the full-order solution from ROM coefficients
        # Combines: full_solution = mean_field + ROM_basis * ROM_coefficients
        u_rec = reconstruct_solution(u_red, V_sel, NLS_train_mean)
        
        # Extract parameter values for this snapshot (thermal conductivity, boundary condition)
        kw = assemble_kwargs(u_rec, training_params[i])
        
        kw.update(extra_kwargs or {})

        # Apply hyperreduced residual operator to get reduced residual evaluation
        # This evaluates the nonlinear residual at a subset of quadrature points
        # prev: previous solution state, k_param: thermal conductivity, q_param: heat source
        q = Residual.hyperreduction(**kw).T.copy()
        
        # Stack residual evaluations from all snapshots
        if q_mus is None:
            # Initialize with first snapshot's residual
            q_mus = q
        else:
            # Concatenate new snapshot residual "on top" of existing array (axis=0)
            q_mus = np.concatenate((q_mus, q), axis=0)
    
    return q_mus



def select_weights(element_indices, weights, mesh):
    element_to_gauss_weights = {}

    # Populate each selected element with weights
    for idx, weight in zip(element_indices, weights):
        # Ensure idx is a scalar by extracting the first element if it's an array
        if isinstance(idx, np.ndarray) and idx.size == 1:
            idx = idx.item()  # Convert single-element array to scalar
        
        if idx not in element_to_gauss_weights:
            # Initialize the element with zero if not already in the dictionary
            element_to_gauss_weights[idx] = 0
        
        # Update the weight for the selected Gauss point
        element_to_gauss_weights[idx] += weight

    all_elems = np.concatenate([
        np.atleast_1d(v).ravel()
        for v in element_to_gauss_weights.keys()
    ])
    weights = np.any(np.isin(mesh.t.T, all_elems), axis = 1)
    weights = weights.astype(int)

    return weights




def compute_nonlinear_snapshots(
    non_linear_func ,
    fos_solutions,
    param_list,
) -> np.ndarray:
    """
    Evaluate a nonlinear RHS function over a set of FOM snapshots.

    Parameters
    ----------
    non_linear_func
        A function with signature non_linear_func(u=<solution>, param=<param>) → array.
    fos_solutions
        Sequence of full-order solutions (each an ndarray).
    param_list
        Sequence of parameter values, same length as fos_solutions.

    Returns
    -------
    F_nl : ndarray
        Array of shape (n_snapshots, ...) where each slice F_nl[i] is
        non_linear_func(u=fos_solutions[i], param=param_list[i]).
    """
    F_nl = []
    for u, param in zip(fos_solutions, param_list):
        rhs_snapshot = non_linear_func(u=u, param=param)
        F_nl.append(np.copy(rhs_snapshot))
    return np.asarray(F_nl)