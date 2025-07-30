import numpy as np                          # core numerical arrays
from skfem import BilinearForm             # decorator for bilinear forms
from skfem.helpers import grad, dot        # gradient and inner-product ops
from properties import k, q               # conductivity and source functions

@BilinearForm
def J_form(du, v, p):
    """
    Assemble Jacobian bilinear form for nonlinear heat conduction.

    Weak form
    ---------
    J(du, v) = ∫ [ k(u) ∇du·∇v
                 + dk(u)·du·⟨∇u,∇v⟩
                 − dq(u)·du·v ] dx

    Parameters
    ----------
    du : array_like or callable
        Trial function increment at quadrature points.
    v : array_like or callable
        Test function at quadrature points.
    p : dict
        Assembly context:
        - u_prev       : ndarray, current solution u
        - k_param      : float, conductivity offset
        - q_param      : float, source offset
        - global_mask  : dict of region masks
        - elem_indices : ndarray of int, optional subset indices

    Returns
    -------
    array_like
        Elementwise contributions for the Jacobian.
    """
    # unpack current state and parameters
    u            = p['u_prev']
    k_param      = p['k_param']
    q_param      = p['q_param']
    mask_dict    = p['global_mask']
    elems        = p.get('elem_indices', None)

    # evaluate conductivity and its derivative
    k_val, dk_val = k(u, k_param,
                      global_mask=mask_dict,
                      elem_indices=elems)
    # evaluate source derivative (we ignore q_val here)
    _, dq_val     = q(u, q_param,
                      global_mask=mask_dict,
                      elem_indices=elems)

    # return the bilinear integrand
    return (
        k_val  * dot(grad(du), grad(v))
      + dk_val * du * dot(grad(u), grad(v))
      - dq_val * du * v
    )