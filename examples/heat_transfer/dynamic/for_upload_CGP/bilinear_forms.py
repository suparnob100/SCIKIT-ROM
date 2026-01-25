from skfem.assembly import BilinearForm
from skfem.helpers import dot, grad
from properties import material_properties
from params import simulation_params

rho, cp, k, T_infty, h, Rboltz, emiss = material_properties()
dt, _,_, _, theta,_ = simulation_params()

### Regular bilinear forms ###
# The bilinear forms are defined in the same way as the weak form of the PDE.


@BilinearForm
def mass_form_bil(u, v, p):
    # Weight the mass by rho*cp.
    return (rho * cp) * u * v


@BilinearForm
def laplace_form_bil(u, v, p):
    return k * dot(grad(u), grad(v))


@BilinearForm
def jacobian(u, v, p):
    # Weighted temperature for nonlinear terms

    return (
        (rho * cp / dt) * u * v  # Time derivative term
        + k * (1 - theta) * dot(grad(u), grad(v))  # Conduction term
    )


### Boundary bilinear forms ###
# The boundary bilinear forms are defined in the same way as the weak form of the PDE.
# The boundary bilinear forms are used to apply the boundary conditions to the PDE.

@BilinearForm
def jac_bnd(u, v, p):
    
    """
    J_bnd = -∫ [ 4 * σ * ε * (T')^3 (1 - θ) * u * v + h (1 - θ) * u * v ] dΓ

    with T' = θ * T_{prev} + (1 - θ) * u  (the "weighted" temperature).
    """
    # Weighted temperature T' at quadrature points:
    #    T_prev is an array over nodes, we evaluate it "element by element."
    #    For consistent usage in a real solver, ensure you're passing
    #    the current guess "u" from your Newton iteration as well.
    u_prev = p['prev']
    elem_indices = p.get('elem_indices', None)

    if elem_indices is not None:
        u_rhs = theta*u_prev[elem_indices] + (1 - theta)*u

    else:
        u_rhs = theta*u_prev + (1 - theta)*u

    
    # Derivative of σ*(T')^4  => 4σ(T')^3, multiplied by emissivity
    rad_term = 4.0 * Rboltz * emiss * (u_rhs**3)

    return (
        (rad_term * (1 - theta) * u * v)  # radiation derivative
        + (h * (1 - theta) * u * v)        # convection derivative
    )


@BilinearForm
def jac_bnd_only_rad(u, v, p):
    
    """
    J_bnd = -∫ [ 4 * σ * ε * (T')^3 (1 - θ) * u * v + h (1 - θ) * u * v ] dΓ

    with T' = θ * T_{prev} + (1 - θ) * u  (the "weighted" temperature).
    """
    # Weighted temperature T' at quadrature points:
    #    T_prev is an array over nodes, we evaluate it "element by element."
    #    For consistent usage in a real solver, ensure you're passing
    #    the current guess "u" from your Newton iteration as well.
    u_prev = p['prev']
    elem_indices = p.get('elem_indices', None)

    if elem_indices is not None:
        u_rhs = theta*u_prev[elem_indices] + (1 - theta)*u

    else:
        u_rhs = theta*u_prev + (1 - theta)*u

    
    # Derivative of σ*(T')^4  => 4σ(T')^3, multiplied by emissivity
    rad_term = 4.0 * Rboltz * emiss * (u_rhs**3)

    return rad_term * (1 - theta) * u * v  # radiation derivative