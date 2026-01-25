from skfem import *
from skfem.assembly import LinearForm
from skfem.helpers import dot, grad
import numpy as np
from properties import material_properties
from params import simulation_params

rho, cp, k, T_infty, h, Rboltz, emiss = material_properties()
dt, _,_, _, theta,_ = simulation_params()

### Regular bilinear forms ###
# The bilinear forms are defined in the same way as the weak form of the PDE.

@LinearForm
def mass_form(v, p):
    # Weight the mass by rho*cp.
    u_ = p['trial']
    u_prev = p['prev']
    return (rho * cp) * (u_-u_prev) * v/dt


@LinearForm
def laplace_form(v, p):
    u_ = p['trial']
    u_prev = p['prev']

    u_rhs = theta*u_prev + (1 - theta)*u_
    u_rhs_grad = theta*grad(u_prev) + (1 - theta)*grad(u_)
    return k * dot(u_rhs_grad, grad(v))


### Boundary bilinear forms ###
# The boundary bilinear forms are defined in the same way as the weak form of the PDE.
# The boundary bilinear forms are used to apply the boundary conditions to the PDE.

@LinearForm
def convection_radiation_bdd(v, p):
    
    u_ = p['trial']
    u_prev = p['prev']
    elem_indices = p.get('elem_indices', None)

    if elem_indices is not None:
        u_rhs = theta*u_prev + (1 - theta)*u_
        u_rhs = u_rhs[elem_indices]

    else:
        u_rhs = theta*u_prev + (1 - theta)*u_

    return h * (u_rhs-T_infty) * v + Rboltz * emiss * (u_rhs**4-T_infty**4 )* v



@LinearForm
def convection_bdd(v, p):
    return -h *T_infty * v


@LinearForm
def radiation_bdd(v, p):
    
    u_ = p['trial']
    u_prev = p['prev']
    elem_indices = p.get('elem_indices', None)

    if elem_indices is not None:
        u_rhs = theta*u_prev + (1 - theta)*u_
        u_rhs = u_rhs[elem_indices]

    else:
        u_rhs = theta*u_prev + (1 - theta)*u_

    return Rboltz * emiss * (u_rhs**4-T_infty**4 )* v


def make_laser_flux(traj, Qp, eta, r):
    @LinearForm
    def laser_flux(v, w):
        t = w['time']

        # laser center at time t
        x_center, y_center = traj.position(t)

        x, y = w.x[0], w.x[1]  # quadrature point coords

        dist_sq = (x - x_center)**2 + (y - y_center)**2
        decay   = np.exp(-2 * dist_sq / r**2)
        factor  = 2 * Qp * eta / (np.pi * r**2)

        return factor * decay * v

    return laser_flux


## Nonliner term to be hyperreduced via ECSW
@LinearForm
def radiation_bdd_hyp(v, p):
    
    u = p['sol']

    return Rboltz * emiss * (u**4-T_infty**4)* v


def residual_fn(u_cur,u_old, bc_laser, M_bi_lin, K_bi_lin, fbasis_non_dirichlet, theta, dt):

    M_lin = M_bi_lin.dot((u_cur - u_old) / dt)
    K_lin = K_bi_lin.dot(theta * u_old + (1 - theta) * u_cur)
    u_prev_bdd  = fbasis_non_dirichlet.interpolate(u_old)
    u_curr_bdd = fbasis_non_dirichlet.interpolate(u_cur)
    bc_neum = asm(convection_radiation_bdd,
                    fbasis_non_dirichlet,
                    trial=u_curr_bdd,
                    prev=u_prev_bdd)
    
    residual   = M_lin + K_lin - bc_neum - bc_laser

    return np.linalg.norm(residual)