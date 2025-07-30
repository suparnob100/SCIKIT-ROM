import numpy as np
from skfem import *
from skfem.helpers import ddot, sym_grad, eye, trace
from properties import lame_params

"""
Affine bilinear forms for 3-D linear elasticity.

Defines BilinearForm components for Lamé parameters:
- stiffness_lam: λ term
- stiffness_mu: μ term
"""

def C_lam(T, lam):
    """
    Compute the λ-scaled isotropic elasticity tensor action.

    Parameters
    ----------
    T : array_like
        Strain tensor (∇sym u).
    lam : float
        First Lamé parameter (λ).

    Returns
    -------
    array_like
        λ * trace(T) * I tensor of appropriate size.
    """
    return lam * eye(trace(T), T.shape[0])


def C_mu(T, mu):
    """
    Compute the μ-scaled part of the elasticity tensor action.

    Parameters
    ----------
    T : array_like
        Strain tensor (∇sym u).
    mu : float
        Second Lamé parameter (μ).

    Returns
    -------
    array_like
        2 * μ * T tensor.
    """
    return 2.0 * mu * T


@BilinearForm
def stiffness_lam(u, v, w):
    """
    λ-term in the linear elasticity bilinear form.

    Parameters
    ----------
    u : array_like
        Trial (displacement) function.
    v : array_like
        Test (displacement) function.
    w : dict
        Assembly kwargs with keys:
        - 'E' : Young's modulus
        - 'nu': Poisson's ratio
        - 'region': region identifier

    Returns
    -------
    float
        Stiffness contribution: λ * (tr(ε(u)) · tr(ε(v))).
    """
    E = w.get('E', None)
    nu = w.get('nu', None)
    region = w.get('region', None)
    if E is None or nu is None:
        lam = 1.0
    else:
        lam, _ = lame_params(E, nu, region)
    return ddot(C_lam(sym_grad(u), lam), sym_grad(v))


@BilinearForm
def stiffness_mu(u, v, w):
    """
    μ-term in the linear elasticity bilinear form.

    Parameters
    ----------
    u : array_like
        Trial (displacement) function.
    v : array_like
        Test (displacement) function.
    w : dict
        Assembly kwargs with keys:
        - 'E' : Young's modulus
        - 'nu': Poisson's ratio
        - 'region': region identifier

    Returns
    -------
    float
        Stiffness contribution: 2μ ⟨ε(u), ε(v)⟩.
    """
    E = w.get('E', None)
    nu = w.get('nu', None)
    region = w.get('region', None)
    if E is None or nu is None:
        mu = 1.0
    else:
        _, mu = lame_params(E, nu, region)
    # 2μ * strain(u) : sym_grad(v)
    return ddot(C_mu(sym_grad(u), mu), sym_grad(v))