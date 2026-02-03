"""\
Bilinear forms for 2-D transient heat conduction.

We use:
- Mass form: m(u, v) = \int u v \, d\Omega
- Diffusion form: a(u, v) = \int \nabla u \cdot \nabla v \, d\Omega

The semi-discrete system is:
    M dT/dt + k K T = q f
"""

from skfem import BilinearForm
from skfem.helpers import dot, grad


@BilinearForm
def a_mass(u, v, w):
    return u * v


@BilinearForm
def a_diffusion(u, v, w):
    return dot(grad(u), grad(v))
