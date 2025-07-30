import numpy as np 

# -----------------------------------------------------------------------------
# Material property functions for thermal conductivity (k) and heat source (q)
# -----------------------------------------------------------------------------

def k(k_param):
    """
    Thermal conductivity parameterized by a flat offset.

    Parameters:
    -----------
    k_param : float
        User-specified conductivity parameter to adjust baseline conductivity.

    Returns:
    --------
    float
        The thermal conductivity, computed as a constant baseline plus k_param.
    """
    # Baseline conductivity (homogeneous material) is 16.0
    return 16.0 + k_param


def q(q_param):
    """
    Heat source term parameterized by a flat offset.

    Parameters:
    -----------
    q_param : float
        User-specified source parameter to adjust baseline heat load.

    Returns:
    --------
    float
        The heat source value, computed as a large constant baseline plus q_param.
    """
    # Baseline source load is 35000.0
    return 35000.0 + q_param
