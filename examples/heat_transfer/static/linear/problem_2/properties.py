def k(k_param, region):
    """
    Piecewise thermal conductivity.

    Parameters
    ----------
    k_param : float
        Conductivity offset.
    region : {'region_1', 'region_2'}
        Region identifier.

    Returns
    -------
    float
        Conductivity value for the specified region.
    """
    if region == 'region_1':
        return 16 + k_param
    else:
        return 30 + k_param


def q(q_param, region):
    """
    Piecewise heat source term.

    Parameters
    ----------
    q_param : float
        Source offset.
    region : {'region_1', 'region_2'}
        Region identifier.

    Returns
    -------
    float
        Source term value for the specified region.
    """
    if region == 'region_1':
        return 35000 + q_param
    else:
        return 10 * q_param + 5000