"""
Parameter sampling for the two-segment beam vibration example.

We vary the segment thickness values:
- h1: thickness of region_1
- h2: thickness of region_2

All other quantities are fixed in properties.py.
"""

import numpy as np
from skrom.rom.rom_utils import generate_sobol


def parameters(N_snap: int = 32):
    """
    Generate Sobol samples for (h1, h2).

    Returns
    -------
    params : ndarray, shape (2*N_snap, 2)
        Stacked train+test samples.
    param_ranges : list[tuple[float,float]]
        Parameter bounds.
    train_mask : ndarray[bool]
    test_mask : ndarray[bool]
    """
    # thickness bounds (meters if using SI)
    h1 = (0.010, 0.030)
    h2 = (0.010, 0.030)
    param_ranges = [h1, h2]

    params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    params_test  = generate_sobol(len(param_ranges), N_snap, param_ranges)

    params = np.vstack([params_train, params_test])

    n_total = len(params)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:N_snap] = True
    test_mask = ~train_mask

    return params, param_ranges, train_mask, test_mask
