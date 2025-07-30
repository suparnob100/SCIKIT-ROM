from skfem import LinearForm

@LinearForm
def l(v, w):
    """
    Template for your problem’s linear (load or residual) form:
        ∫_Ω [your integrand here] dx

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
    # TODO: define your integrand, and extract your params as e.g. coeff = w.get("coeff", 1.0)
