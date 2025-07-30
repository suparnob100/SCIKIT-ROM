import numpy as np                        # numerical arrays
from .properties import rho, E         # conductivity/source functions
from skfem import LinearForm              # decorator for residual forms
from skfem.helpers import grad, dot

def fx(w):
    L = 420
    A = 10
    rho_param = w['rho_param']
    rho_val = rho(rho_param)

    return 1000 + rho_val * A * (L - w.x[0])


@LinearForm
def R(v, w):
    """
    Residual ⟨R(u), v⟩ = ∫ EA |u'|^{n-1} u' v' − ∫ f v
    u : current FE field (TrialFunction inside scikit-fem)
    w : TestFunction
    """
    n_exp = 0.26
    A = 10

    u_prev  = w['u_prev']
    mu = w['mu']

    E_val = E(mu)

    G = E_val * A * (np.abs(grad(u_prev)))**(n_exp-1) * grad(u_prev)
    return dot(G, grad(v)) - fx(w) * v