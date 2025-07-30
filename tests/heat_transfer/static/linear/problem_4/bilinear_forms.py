import numpy as np
from skfem import BilinearForm
from skfem.helpers import grad, dot

@BilinearForm
def a(T, v, w):
    """
    Stiffness bilinear form for 1-D heat conduction.

    Weak form
    ---------
    ∫ Ω k_param · ⟨∇T, ∇v⟩ dx

    Parameters
    ----------
    T : array_like or callable
        Trial function at quadrature points.
    v : array_like or callable
        Test function at quadrature points.
    w : dict
        Assembly context with:
        - 'k_param' : float, conductivity offset
        - 'q_param' : float, unused here

    Returns
    -------
    array_like or float
        Elementwise integrand: k_param * ⟨grad(T), grad(v)⟩.
    """
    k_val = w.get('k_param', 1.0)
    return k_val * dot(grad(T), grad(v))