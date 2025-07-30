import numpy as np                          # core numerical arrays
from skfem import BilinearForm             # decorator for bilinear forms
from properties import  E
from skfem.helpers import grad, dot


@BilinearForm
def J_form(du, v, w):
    """
    Tangent (Jacobian) ⟨J(u) u, v⟩ =
      ∫ E A n |u'|^{n-1} u' v' dx
    """
    # problem constants
    A_val = 10.0
    n_exp = 0.26

    # physical coordinate if you ever need it (not needed here)
    # x = w.x[0]
    u_prev = w['u_prev']
    mu = w['mu']
    # evaluate your parameter-dependent fields
    E_val   = E(mu)            # broadcast similarly

    # current solution gradient at quad points

    # assume g = grad(u_prev)  # shape = (dim, nqp)
    g = grad(u_prev) 
    # 1) Clamp large gradients to ±g_max
        
    # 1) raw FE gradient (shape: dim×nqp)
    
    # 2) clamp large gradients to a max magnitude g_max
    g_max     = 1e8
    g_clamped = np.clip(g, -g_max, g_max)
    
    # 3) regularize tiny gradients to avoid 0^negative
    eps        = 1e-8
    abs_g_safe= np.maximum(np.abs(g_clamped), eps)
    
    # build the integrand: E*A * n * |ux|^(n-1) * v.grad * w.grad
    G = E_val * A_val * n_exp * (abs_g_safe)**(n_exp-1) * grad(du)
    # print(grad(u_prev))
    return dot(G, grad(v))
    

# @BilinearForm
# def K_form(du, v, w):
#     """
#     Tangent (Jacobian) ⟨J(u) u, v⟩ =
#       ∫ E A n |u'|^{n-1} u' v' dx
#     """
#     # problem constants
#     A_val = 10.0
#     n_exp = 0.26

#     # physical coordinate if you ever need it (not needed here)
#     # x = w.x[0]
#     u_prev = w['u_prev']
#     mu = w['mu']
#     # evaluate your parameter-dependent fields
#     E_val   = E(mu)            # broadcast similarly

#     # current solution gradient at quad points

#     # build the integrand: E*A * n * |ux|^(n-1) * v.grad * w.grad
#     #return E_val * A_val * (grad(u_prev)**(n_exp-1)) * dot(grad(du), grad(v))
