from properties import q        # heat source function
from skfem import LinearForm
import numpy as np

@LinearForm
def l(v, w):
    """
    Linear form for the heat equation’s source term.

    Weak form
    ---------
    ∫ Ω q(q_param, x) · v(x) dx

    Parameters
    ----------
    v : array_like
        Test function values at quadrature points.
    w : dict
        Assembly context:
        - q_param : float, heat source parameter
        - x       : array_like, quadrature point coordinates

    Returns
    -------
    array_like or float
        Elementwise integrand q(q_param, coords) * v.
    """
    q_param = w.get('q_param')
    if q_param is None:
        rhs = 1
    else:
        coords = w.x
        rhs = q(q_param, coords[0] + coords[1])
    return rhs * v
