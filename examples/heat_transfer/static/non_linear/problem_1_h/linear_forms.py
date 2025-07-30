"""
Nonlinear residual linear form for Newton–Raphson iterations:

    R(v) = ∫ [ k(u) ∇u · ∇v − q(u) v ] dx
"""
import numpy as np                        # numerical arrays
from properties import k, q              # conductivity/source functions
from skfem.helpers import grad, dot       # gradient and dot-product
from skfem import LinearForm              # decorator for residual forms

@LinearForm
def R(v, p):
    """
    Assemble the nonlinear residual R(v) for the current iterate u.

    Parameters
    ----------
    v : array_like
        Test function evaluated at quadrature points.
    p : dict
        Assembly context with keys:
        - 'u_prev'         : ndarray, current solution u
        - 'k_param'        : float, conductivity offset
        - 'q_param'        : float, source offset
        - 'global_mask'    : dict of region masks
        - 'elem_indices'   : optional subset indices

    Returns
    -------
    ndarray
        Elementwise contributions to the residual.
    """
    # unpack solution and parameters
    u       = p['u_prev']
    k_param = p['k_param']
    q_param = p['q_param']
    mask_dict = p['global_mask']
    elems   = p.get('elem_indices', None)

    # evaluate k(u) and q(u) on the masked subset
    k_val, _ = k(u, k_param, global_mask=mask_dict, elem_indices=elems)
    q_val, _ = q(u, q_param, global_mask=mask_dict, elem_indices=elems)

    # residual integrand: conductivity term minus source term
    return k_val * dot(grad(u), grad(v)) - q_val * v


@LinearForm
def rhs(v, p):
    """
    Assemble the nonlinear residual R(v) for the current iterate u.

    Parameters
    ----------
    v : array_like
        Test function evaluated at quadrature points.
    p : dict
        Assembly context with keys:
        - 'u_prev'         : ndarray, current solution u
        - 'k_param'        : float, conductivity offset
        - 'q_param'        : float, source offset
        - 'global_mask'    : dict of region masks
        - 'elem_indices'   : optional subset indices

    Returns
    -------
    ndarray
        Elementwise contributions to the residual.
    """
    # unpack solution and parameters
    u       = p['u_prev']
    k_param = p['k_param']
    q_param = p['q_param']
    mask_dict = p['global_mask']
    elems   = p.get('elem_indices', None)

    # evaluate k(u) and q(u) on the masked subset
    k_val, _ = k(u, k_param, global_mask=mask_dict, elem_indices=elems)
    q_val, _ = q(u, q_param, global_mask=mask_dict, elem_indices=elems)

    # residual integrand: conductivity term minus source term
    return q_val * v