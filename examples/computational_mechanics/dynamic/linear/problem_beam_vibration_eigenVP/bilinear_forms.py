"""
Bilinear forms used to assemble parameter-independent operator blocks.
"""

from skfem import BilinearForm
from skfem.helpers import grad, dot


@BilinearForm
def a_grad(u, v, w):
    """∫ u' v' dx"""
    return dot(grad(u), grad(v))


@BilinearForm
def a_mass(u, v, w):
    """∫ u v dx"""
    return u * v


@BilinearForm
def a_shear_w_theta(theta, v, w):
    """
    Shear coupling (w equation, theta unknown):
    -∫ theta * v' dx
    """
    return -theta * grad(v)[0]


@BilinearForm
def a_shear_theta_w(u, phi, w):
    """
    Shear coupling (theta equation, w unknown):
    -∫ u' * phi dx
    """
    return -grad(u)[0] * phi
