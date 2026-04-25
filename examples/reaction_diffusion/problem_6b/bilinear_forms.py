from skfem import BilinearForm
from skfem.helpers import grad, dot

@BilinearForm
def a_diff(u, v, w):
    """
    Diffusion operator:
        ∫ grad(u) · grad(v) dx
    """
    return dot(grad(u), grad(v))

@BilinearForm
def a_reac(u, v, w):
    """
    Reaction operator:
        ∫ u v dx
    """
    return u * v