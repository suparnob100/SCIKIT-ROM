import numpy as np
from skrom.rom.rom_utils import generate_sobol  # Sobol sampler for low-discrepancy parameter sampling

def parameters(N_snap):
    """
    Template for generating training/testing parameter samples.

    Uses generate_sobol from skrom.rom.rom_utils to sample uniformly
    over each interval in param_ranges.

    Parameters
    ----------
    N_snap : int
        Number of samples per set.

    Returns
    -------
    params : ndarray, shape (2*N_snap, D)
        Stacked [training; testing] samples.
    param_ranges : list of tuple
        [(p1_min, p1_max), ..., (pD_min, pD_max)] for each parameter.
    train_mask : ndarray of bool
        True for the first N_snap entries of params.
    test_mask : ndarray of bool
        True for the remaining entries.
    """
    # TODO: define your parameter intervals, e.g.
    # param_ranges = [(min1, max1), ..., (minD, maxD)]

    # TODO: sample training and testing points
    # params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    # params_test  = generate_sobol(len(param_ranges), N_snap, param_ranges)

    # TODO: merge and build masks
    # params = np.vstack([params_train, params_test])
    # n_total = params.shape[0]
    # train_mask = np.zeros(n_total, dtype=bool)
    # train_mask[:N_snap] = True
    # test_mask = ~train_mask

    return params, param_ranges, train_mask, test_mask