import numpy as np
from skrom.rom.rom_utils import generate_sobol

def parameters(N_snap=32):
    """
    Parameter vector:
        [D1, D2, D3, Sigma1, Sigma2, Sigma3, Q1, Q2, Q3]
    """

    d = 5
    D1_range = (0.1, 1)
    D2_range = (0.1, 1)
    D3_range = (0.1, 1)

    z = 0.0
    a = 2.0 
    Sigma1_range = (a*0, a*2)
    Sigma2_range = (a, a)
    Sigma3_range = (a*0, a*3)

    q = 10.0 
    Q1_range = (q*0, q*0) # obstacle
    Q2_range = (q, q)     # center
    Q3_range = (q*3, q*3) # background

    param_ranges = [
        D1_range, D2_range, D3_range,
        Sigma1_range, Sigma2_range, Sigma3_range,
        Q1_range, Q2_range, Q3_range
    ]

    params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    params_test  = generate_sobol(len(param_ranges), N_snap, param_ranges)

    params = np.vstack([params_train, params_test])

    train_mask = np.zeros(len(params), dtype=bool)
    train_mask[:N_snap] = True
    test_mask = ~train_mask

    return params, param_ranges, train_mask, test_mask