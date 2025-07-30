import numpy as np 

# -----------------------------------------------------------------------------
# Material property functions for thermal conductivity (k) and heat source (q)
# -----------------------------------------------------------------------------

def q(q_param, x):
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
    return q_param * np.sin(x)
