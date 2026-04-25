import numpy as np
from skfem import BilinearForm
from skfem.helpers import ddot, sym_grad, trace

"""
Affine bilinear forms for 2D isotropic linear elasticity (plane stress).

We use an affine split in terms of (lam_bar, mu):

  a(u,v) = lam_bar * a_lam(u,v) + mu * a_mu(u,v)

where (lam_bar, mu) are the effective plane-stress coefficients returned by
properties.lame_params_plane_stress(E, nu).

Important:
- The forms below are parameter-free (assembled once).
- Parameter dependence enters only through the scalar coefficients in problem_def.py.
"""


@BilinearForm
def stiffness_lam(u, v, w):
    # trace(eps(u)) * trace(eps(v))
    return trace(sym_grad(u)) * trace(sym_grad(v))


@BilinearForm
def stiffness_mu(u, v, w):
    # 2 * eps(u) : eps(v)
    return 2.0 * ddot(sym_grad(u), sym_grad(v))
