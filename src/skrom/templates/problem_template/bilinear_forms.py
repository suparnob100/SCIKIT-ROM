"""Template bilinear form definition.

TL;DR
-----
This module shows where a user should define the problem bilinear form or nonlinear Jacobian.

Notes
-----
The starter function is decorated for scikit-fem assembly and returns a placeholder integrand.
"""

from skfem import BilinearForm

@BilinearForm
def a(u, v, w):
    """Template for a problem bilinear form or nonlinear-solver Jacobian:
        ∫_Ω [integrand here] dx
    
    TL;DR
    -----
    Template for a problem bilinear form or nonlinear-solver Jacobian.
    
    Parameters:
    -----------
    u : array_like or callable
        Trial (or current Newton iterate) values at quadrature points.
    v : array_like or callable
        Test function values at quadrature points.
    w : dict
        Assembly context carrying any coefficients or parameters.
    
    Returns:
    --------
    array_like or float
        Elementwise integrand for global matrix assembly.
    
    Note:
    -----
    In a nonlinear problem solved by Newton’s method,
    this form assembles the Jacobian matrix.
    """
    # TODO: define the integrand, and extract parameters as e.g. coeff = w.get("coeff", 1.0)
