"""
problem_plate_with_hole_tension_2d: 2D plane-stress plate-with-hole in tension (Affine)

- PDE: linear elasticity (small strain), plane stress
- BC:
    * Dirichlet (clamp): u = (0,0) on 'left'
    * Neumann (traction): t = (t0, 0) on 'right'
    * Free: top, bottom, and hole boundary
- Parameters: (E, nu)
- ROM: standard POD-Galerkin (no hyperreduction)
"""

from skrom.problem_classes.static.master_class import register_problem, Problem
import numpy as np
import os
from skrom.rom.rom_utils import _ensure_csr
from skrom.fom.fem_utils import load_domain
from skfem import asm, condense, solve

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))

@register_problem(PROBLEM_NAME)
class ProblemAffinePlaneStress(Problem):

    def domain(self):
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        from bilinear_forms import stiffness_lam, stiffness_mu
        return [stiffness_lam, stiffness_mu]

    def linear_forms(self):
        from linear_forms import traction
        return [traction]

    def properties(self):
        from properties import lame_params_plane_stress
        return lame_params_plane_stress

    def parameters(self, n_samples):
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, param):
        # t0 is the traction magnitude used by linear_forms.traction
        return dict(E=param[0], nu=param[1], t0=1.0)

    def fom_operators(self, cls):
        """Assemble and cache affine stiffness blocks."""
        if cls.cur_itr == 0:
            load_domain(self)

            self.stiffnesses = {
                "lam": self.bilinear_forms()[0],
                "mu":  self.bilinear_forms()[1],
            }

            cls.K_list = {
                region: {
                    mat: asm(form, basis).copy()
                    for mat, form in self.stiffnesses.items()
                }
                for region, basis in self.basis_regions.items()
            }
        return cls.K_list

    def fom_rhs(self, cls):
        """Assemble and cache Neumann load vector."""
        if cls.cur_itr == 0:
            load_domain(self)
            b_linear = self.linear_forms()[0]
            cls.rhs_linear = asm(b_linear, self.fbasis_neumann, t0=1.0)
        return cls.rhs_linear

    def fom_solver(self, cls, param):
        """Solve the full-order model (FOM)."""
        self.K_list = self.fom_operators(cls)
        self.rhs_linear = self.fom_rhs(cls)

        E, nu = float(param[0]), float(param[1])

        K = 0.0
        for region, mats in self.K_list.items():
            lam_bar, mu = self.properties()(E, nu, region)
            K = K + lam_bar * mats['lam'] + mu * mats['mu']

        u = self.basis.zeros()
        u[self.dirichlet_dofs] = self.dirichlet_boundary_value

        Kc, fc, x, I = condense(K, self.rhs_linear, x=u, D=self.dirichlet_dofs)
        uc = solve(Kc, fc)
        x[I] = uc
        return x

    def reduced_operators(self, cls):
        """Project affine operators onto reduced basis + mean-shift terms."""
        self.U   = cls.V_sel.copy()
        self.T_k = cls.train_ref.copy()

        self.K_a = {
            region: {mat: _ensure_csr(block) for mat, block in mats.items()}
            for region, mats in self.K_list.items()
        }

        self.K_r_a = {
            region: {mat: (self.U.T @ B) @ self.U for mat, B in sub.items()}
            for region, sub in self.K_a.items()
        }

        self.K_a_Tk = {
            region: {mat: (self.U.T @ B) @ self.T_k for mat, B in sub.items()}
            for region, sub in self.K_a.items()
        }

        self.f_term = self.U.T @ self.rhs_linear

    def rom_solver(self, cls, param):
        """Solve the reduced-order model (ROM) and return reduced coordinates."""
        self.K_list = cls.K_list.item()
        self.rhs_linear = cls.rhs_linear

        if cls.cur_itr == 0:
            self.reduced_operators(cls)

        E, nu = float(param[0]), float(param[1])

        props = {
            region: {
                mat: val
                for mat, val in zip(
                    self.K_list[region].keys(),
                    self.properties()(E, nu, region)
                )
            }
            for region in self.K_list
        }

        self.K_r = sum(
            props[r][mat] * self.K_r_a[r][mat]
            for r in props for mat in props[r]
        )

        self.T_k_term = sum(
            props[r][mat] * self.K_a_Tk[r][mat]
            for r in props for mat in props[r]
        )

        g = self.f_term - self.T_k_term
        u_rom = np.linalg.solve(self.K_r, g)
        return u_rom

    def hyper_rom_solver_deim(self):
        """Not used in this example."""
        pass

    def hyper_rom_solver_ecsw(self):
        """Not used in this example."""
        pass
