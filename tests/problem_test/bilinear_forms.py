from skfem import BilinearForm

@BilinearForm
def a(u, v, w):
    """
    Template for your problem's bilinear form (or Jacobian in a nonlinear solver):
        ∫_Ω [your integrand here] dx

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
    # TODO: define your integrand, and extract your params as e.g. coeff = w.get("coeff", 1.0)