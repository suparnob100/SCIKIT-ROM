import numpy as np  # array operations

def k(u, k_param, global_mask, elem_indices):
    """
    Piecewise thermal conductivity and its derivative.

    Parameters
    ----------
    u : ndarray
        Temperature at DOFs or quadrature points.
    k_param : float
        Base conductivity offset.
    global_mask : dict
        Region masks, e.g. {"region_1": mask1, "region_2": mask2}.
    elem_indices : ndarray of int, optional
        Subset of element indices to apply masks.

    Returns
    -------
    k_val : ndarray
        Conductivity at each point.
    dk_val : ndarray
        ∂k/∂u at each point.
    """
    # restrict masks if needed
    mask = (
        {key: m[elem_indices] for key, m in global_mask.items()}
        if elem_indices is not None
        else dict(global_mask)
    )


    # init outputs
    k_val = np.zeros_like(u)
    dk_val = np.zeros_like(u)

    # unpack region masks
    region_1, region_2 = mask.values()
    u1 = u[region_1]
    u2 = u[region_2]

    # left region: nonlinear formula
    k_val[region_1]  = 16 + k_param + 2150.0 / (u1 - 73.15)
    dk_val[region_1] = -2150.0 / (u1 - 73.15)**2

    # right region: cubic polynomial
    k_val[region_2]  = (
        30 + k_param
        + 2.09e-2 * u2
        - 1.45e-5 * u2**2
        + 7.67e-9 * u2**3
    )
    dk_val[region_2] = (
        2.09e-2
        - 2 * 1.45e-5 * u2
        + 3 * 7.67e-9 * u2**2
    )

    return k_val, dk_val


def q(u, q_param, global_mask, elem_indices):
    """
    Piecewise heat source and its derivative.

    Parameters
    ----------
    u : ndarray
        Temperature at DOFs or quadrature points.
    q_param : float
        Base source offset.
    global_mask : dict
        Region masks, e.g. {"region_1": mask1, "region_2": mask2}.
    elem_indices : ndarray of int, optional
        Subset of element indices to apply masks.

    Returns
    -------
    q_val : ndarray
        Source term at each point.
    dq_val : ndarray
        ∂q/∂u at each point.
    """
    # restrict masks if needed
    mask = (
        {key: m[elem_indices] for key, m in global_mask.items()}
        if elem_indices is not None
        else dict(global_mask)
    )

    # init outputs
    q_val  = np.zeros_like(u)
    dq_val = np.zeros_like(u)

    # unpack region masks
    region_1, region_2 = mask.values()
    u1 = u[region_1]
    u2 = u[region_2]

    # left region: linear + offset
    q_val[region_1] = q_param + 35000.0 + u1 / 10.0
    dq_val[region_1] = 0.1

    # right region: constant source
    q_val[region_2] = 10 * q_param + 5000.0
    dq_val[region_2] = 0.0

    return q_val, dq_val