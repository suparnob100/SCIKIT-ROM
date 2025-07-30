import numpy as np  # Numerical computing library for array operations
import sys
sys.path.append("src")     # Ensure project root 'src' folder is in PYTHONPATH
from src.skrom.rom.rom_utils import generate_sobol  # Sobol sequence sampler

def parameters(N_snap=32):
    """
    Generate Sobol-based training and testing parameter samples for ROM/FE simulations.

    Parameters:
    -----------
    N_snap : int, default=32
        Number of Sobol samples per parameter set for training (and equal count for testing).

    Returns:
    --------
    params : ndarray, shape (2*N_snap, 2)
        Stack of [k_param, q_param] samples: first N_snap training, then N_snap testing.
    param_ranges : list of tuple
        [(k_min, k_max), (q_min, q_max)] ranges used for scaling the Sobol samples.
    train_mask : ndarray of bool, shape (2*N_snap,)
        True for training-sample indices, False otherwise.
    test_mask : ndarray of bool, shape (2*N_snap,)
        True for testing-sample indices, False otherwise.

    Notes:
    -----
    - Uses a Sobol low-discrepancy sequence to uniformly sample the 2D parameter space.
    - Training and testing sets are drawn independently but from the same uniform range.
    """
    # ------------------------------------------------------------
    # 1) Define parameter intervals for conductivity (k) and source (q)
    # ------------------------------------------------------------
    q_param = (0.2, 6)
    param_ranges = [q_param]

    # ------------------------------------------------------------
    # 2) Sample Sobol points in the unit hypercube then scale to each param range
    # ------------------------------------------------------------
    params_train = generate_sobol(
        len(param_ranges),  # 2D: [k, q]
        N_snap,
        param_ranges
    )
    params_test = generate_sobol(
        len(param_ranges),
        N_snap,
        param_ranges
    )

    # ------------------------------------------------------------
    # 3) Merge training + testing arrays
    # ------------------------------------------------------------
    params = np.vstack([params_train, params_test])

    # ------------------------------------------------------------
    # 4) Build boolean masks to index training vs. testing
    # ------------------------------------------------------------
    n_total = len(params)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:N_snap] = True
    test_mask  = ~train_mask  # complement: True for indices N_snap:2*N_snap

    return params, param_ranges, train_mask, test_mask
