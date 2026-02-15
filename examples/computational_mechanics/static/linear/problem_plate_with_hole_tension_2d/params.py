import numpy as np
from skrom.rom.rom_utils import generate_sobol

def parameters(N_snap=32):
    """
    Parameter sampling for (E, nu).

    Returns
    -------
    params : ndarray (2*N_snap, 2)
    param_ranges : list of tuples
    train_mask, test_mask : boolean masks
    """
    E  = (2.0, 10.0)
    nu = (0.20, 0.45)
    param_ranges = [E, nu]

    params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    params_test  = generate_sobol(len(param_ranges), N_snap, param_ranges)

    params = np.append(params_train, params_test, axis=0)

    train_mask = np.ones(len(params), dtype=bool)
    train_mask[len(params_train):] = False

    test_mask = np.zeros(len(params), dtype=bool)
    test_mask[len(params_train):] = True

    return params, param_ranges, train_mask, test_mask
