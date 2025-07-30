import numpy as np  # Fundamental package for numerical computing
from .properties import q   # Local property function: heat source term

from skfem import (
    LinearForm,
)

# -----------------------------------------------------------------------------
# Linear form definition for the FEM right-hand side: \l(v) = ∫ q * v dx
# -----------------------------------------------------------------------------

@LinearForm
def l(v, w):
    """
    Assemble the linear form for the heat equation's source term.

    Computes the integral:
        ∫_Ω q_param_based_source * v(x) dx
    on each element, where q(x) ≡ q(q_param) is piecewise constant.

    Parameters:
    -----------
    v : array_like
        Values of the test/trial function at quadrature points.
    w : dict
        Assembly context carrying parameter values:
          - w['q_param'] : float
          - w['k_param'] : float (unused here, but provided globally)

    Returns:
    --------
    ndarray or float
        Elementwise product q(q_param) * v, ready for integration by the assembler.
    """
    # Retrieve the user-specified forcing parameter from the assembly context
    q_param = w.get('q_param', None)

    if q_param is None:
        heat_source = 1
    else:
        heat_source = q(q_param)

    return heat_source * v

