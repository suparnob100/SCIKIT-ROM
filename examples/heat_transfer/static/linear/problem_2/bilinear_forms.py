import numpy as np  
from skfem import (
    BilinearForm,   
)
from skfem.helpers import grad, dot  
from properties import k  

@BilinearForm
def a(T, v, w):
    """
    Assemble the stiffness bilinear form for 1-D heat conduction.

    Weak form:
        k * ∫_Ω (dT/dx) * (dv/dx) dx

    Parameters:
    -----------
    T : array_like or callable
        Trial function (or its gradient) values at quadrature points.
    v : array_like or callable
        Test function (or its gradient) values at quadrature points.
    w : dict
        Assembly context carrying parameter values:
            w['k_param'] : float  # conductivity offset
            w['q_param'] : float  # not used here

    Returns:
    --------
    array_like or float
        Elementwise integrand: conductivity * ⟨grad(T), grad(v)⟩.
    """
    # Retrieve conductivity parameter from context (default to baseline if missing)
    k_param = w.get('k_param', None)

    # Evaluate thermal conductivity for this element
    if k_param is None:
        conductivity = 1
    else:
        conductivity = k(k_param)

    # Compute and return the integrand for stiffness assembly
    return conductivity * dot(grad(T), grad(v))

