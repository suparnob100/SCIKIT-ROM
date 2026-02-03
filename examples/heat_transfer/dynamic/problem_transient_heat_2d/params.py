"""\
Parameter sampling for the 2-D transient heat conduction example.

We sample two parameters:
- k in [0.1, 5.0]
- q in [0.0, 5.0]

Sampling uses Sobol points to match other examples.
"""

import numpy as np
from skrom.rom.rom_utils import generate_sobol


def parameters(N_snap: int = 32):
    k_param = (0.1, 5.0)
    q_param = (0.0, 5.0)
    param_ranges = [k_param, q_param]

    params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    params_test = generate_sobol(len(param_ranges), N_snap, param_ranges)

    params = np.vstack([params_train, params_test])

    n_total = len(params)
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:N_snap] = True
    test_mask = ~train_mask

    return params, param_ranges, train_mask, test_mask
