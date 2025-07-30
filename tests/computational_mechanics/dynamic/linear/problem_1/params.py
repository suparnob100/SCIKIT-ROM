import numpy as np  
from src.skrom.rom.rom_utils import generate_sobol  # Sobol sequence generator for parameter sampling


def parameters(N_snap=32):
    """
    Generate training and testing parameter sets using Sobol sequences.

    Parameters:
    -----------
    N_snap : int, default 32
        Number of samples for training and testing each.

    Returns:
    --------
    params : ndarray, shape (2*N_snap, 2)
        Combined array of (k_param, q_param) samples for training followed by testing.
    train_mask : ndarray of bool, shape (2*N_snap,)
        Boolean mask with True for training samples, False for testing samples.
    test_mask : ndarray of bool, shape (2*N_snap,)
        Boolean mask with False for training samples, True for testing samples.
    """
    # Define parameter ranges: (min, max) for k_param and q_param
    nu = (0.2, 0.4)
    E = (3,10)
    param_ranges = [E, nu]
    
    # Generate Sobol sequences for training and testing
    params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    params_test = generate_sobol(len(param_ranges), N_snap, param_ranges)

    # Combine training and testing samples into one array
    params = np.append(params_train, params_test, axis=0)

    # Create masks to separate training and testing sets
    train_mask = np.ones(len(params), dtype=bool)
    train_mask[len(params_train):] = False

    test_mask = np.zeros(len(params), dtype=bool)
    test_mask[len(params_train):] = True

    return params, param_ranges, train_mask, test_mask
