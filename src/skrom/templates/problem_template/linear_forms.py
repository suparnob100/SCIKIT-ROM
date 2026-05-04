"""Template linear form definition.

TL;DR
-----
This module shows where a user should define the problem load or residual form.

Notes
-----
The starter function is decorated for scikit-fem assembly and returns a placeholder integrand.
"""

from skfem import LinearForm

@LinearForm
def l(v, w):
    """Template for a problem linear, load, or residual form:
        ∫_Ω [integrand here] dx
    
    TL;DR
    -----
    Template for a problem linear, load, or residual form.
    
    Parameters:
    -----------
    v : array_like or callable
        Test function values at quadrature points.
    w : dict
        Assembly context carrying any coefficients or parameters.
    
    Returns:
    --------
    array_like or float
        Elementwise integrand for right-hand side vector assembly.
    
    Note:
    -----
    In a nonlinear problem solved by Newton’s method,
    this form assembles the residual vector.
    """
    # TODO: define the integrand, and extract parameters as e.g. coeff = w.get("coeff", 1.0)
