import numpy as np
from skfem import LinearForm

"""
Module defining affine linear form for Neumann traction in 3-D elasticity.
"""

@LinearForm
def traction(v, w):
    """
    Assemble Neumann boundary traction term.

    Parameters
    ----------
    v : array_like
        Test function values at quadrature points.
    w : dict
        Assembly kwargs (unused here).

    Returns
    -------
    float
        Traction contribution scaled by prescribed magnitude.
    """
    # Apply constant traction of magnitude 1e-2 in the y-direction
    return -1e-2 * v[1]