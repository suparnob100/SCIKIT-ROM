import numpy as np
from skrom.rom.rom_utils import generate_sobol  # Sobol sampler for low-discrepancy parameter sampling
from laser_trajectory import *

def parameters(N_snap=5):
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

    q_test = (200,800)
    # # TODO: define your parameter intervals, e.g.
    param_ranges = [q_test]

    # # TODO: sample training and testing points
    # params_train = generate_sobol(len(param_ranges), N_snap, param_ranges)
    # params_test  = generate_sobol(len(param_ranges), 2*N_snap, param_ranges)#[0:1,:]

    params_train = np.array([[200.        ],
       [285.71428571],
       [371.42857143],
       [457.14285714],
       [542.85714286],
       [628.57142857],
       [714.28571429],
       [800.        ],
       [500.]])

    # params_train = np.array([[357.88619518],
    #    [545.16385123],
    #    [729.04546466],
    #    [316.32367503],
    #    [203.32190916],
    #    [690.89365602],
    #    [583.33509695],
    #    [470.90514507]])
    

    params_test  = np.array([[475.54363441],
       [698.85874167],
       [546.08182609],
       [319.24994942],
       [269.26176809],
       [642.42974836],
       [789.93665073],
       [354.2538438 ],
       [726.39822159],
       [593.54204554],
       [215.53943716],
       [297.97226246],
       [519.96951103],
       [687.39715293],
       [459.54167377]])

    # TODO: merge and build masks
    params = np.vstack([params_train, params_test])
    n_total = params.shape[0]
    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:len(params_train)] = True
    test_mask = ~train_mask

    return params, param_ranges, train_mask, test_mask


def simulation_params():

    dt = 0.01  # time step size (s)
    nsteps = 300
    t_end = nsteps*dt
    ts = np.arange(0., t_end + dt/2, dt)
    theta = 0.5      # Crank–Nicolson method (theta=0.5)


    cases = [
    #1
    {
        'lx':[40.0], 'ly':[10.0], 'lz':[6.0], 'factor':[4],
        'eta':[0.4], 'feed_rate':[10], 'r':[1.5],
        'traj_class':[AnyStraightLineTrajectory],
        'traj_kwargs':[{'x0': 1.0, 'y0': 5, 'direction_x': 10.0, 'direction_y': 0}]
    },
    ]
# 'Qp':[250.0],
    return (dt, nsteps, t_end, ts, theta, cases)



