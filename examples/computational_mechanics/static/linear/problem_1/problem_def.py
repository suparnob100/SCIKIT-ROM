"""
Problem_1: Linear Elasticity in a static setting (Affine)
"""
from skrom.problem_classes.static.master_class import register_problem, Problem
import numpy as np
import os
from skrom.rom.rom_utils import _ensure_csr
from skrom.fom.fem_utils import load_domain
from skfem import asm, condense, solve

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))
# global_mask = None

@register_problem(PROBLEM_NAME)
class ProblemAffine(Problem):
    """
    Reduced-order elasticity problem with affine decomposition for Lamé parameters.

    Attributes
    ----------
    basis_regions : dict
        Maps region names to FE basis objects.
    K_list : dict
        Affine stiffness blocks by region/material.
    rhs_linear : ndarray
        Assembled Neumann load vector.
    """

    def domain(self):
        """
        Load geometry, mesh, basis, and boundary DOFs.

        Returns
        -------
        dict
            Domain description from domain_.py.
        """
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        """
        Collect affine bilinear forms for λ and μ.

        Returns
        -------
        [stiffness_lam, stiffness_mu]
        """
        from bilinear_forms import stiffness_lam, stiffness_mu
        return [stiffness_lam, stiffness_mu]

    def linear_forms(self):
        """
        Collect affine linear form for traction.

        Returns
        -------
        [traction]
        """
        from linear_forms import traction
        return [traction]

    def properties(self):
        """
        Return function mapping (E, ν, region) → (λ, μ).

        Returns
        -------
        callable
            lame_params(E, ν, region)
        """
        from properties import lame_params
        return lame_params

    def parameters(self, n_samples):
        """
        Generate (E, ν) samples for offline stage.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        tuple
            Sample arrays and train/test masks.
        """
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, param):
        """
        Pack E, ν into kwargs for assembly routines.

        Parameters
        ----------
        param : tuple (E, ν)

        Returns
        -------
        dict
            {'E': E, 'nu': ν}
        """
        return dict(E=param[0], nu=param[1])

    def fom_operators(self, cls):
        """
        Assemble or reuse region-wise stiffness/load blocks.

        On first iteration, caches:
          - cls.K_list   : {region: {mat: K_block}}
          - cls.rhs_linear : global Neumann vector
        """
        if cls.cur_itr == 0:
            load_domain(self)  # init mesh, basis, DOFs

            # map material keys to forms
            self.stiffnesses = {
                "lam": self.bilinear_forms()[0],
                "mu":  self.bilinear_forms()[1]
            }

            # assemble per-region, per-material blocks
            cls.K_list = {
                region: {
                    mat: asm(form, basis).copy()
                    for mat, form in self.stiffnesses.items()
                }
                for region, basis in self.basis_regions.items()
            }


        return cls.K_list
    
    def fom_rhs(self, cls):
        if cls.cur_itr == 0:
            load_domain(self)  # init mesh, basis, DOFs
            b_linear = self.linear_forms()[0]
            cls.rhs_linear = asm(b_linear, self.fbasis_neumann)

        return cls.rhs_linear


    def fom_solver(self, cls, param):
        """
        Solve the full-order elasticity system.

        Parameters
        ----------
        param : (E, ν)

        Returns
        -------
        u_sol : ndarray
            Displacement field satisfying Dirichlet BCs.
        """
        # retrieve assembled blocks
        self.K_list, self.rhs_linear = self.fom_operators(cls), self.fom_rhs(cls)
        E, nu = param

        # combine affine blocks using computed Lamé coefficients
        K = sum(
            block * coef
            for region, mats in self.K_list.items()
            for block, coef in zip(
                mats.values(),
                self.properties()(E, nu, region)
            )
        )

        # apply Dirichlet BC and solve condensed system
        u = self.basis.zeros()
        u[self.dirichlet_dofs] = self.dirichlet_boundary_value
        return solve(*condense(K, self.rhs_linear, x=u, D=self.dirichlet_dofs))

    def reduced_operators(self, cls):
        """
        Project FOM operators onto reduced basis and compute mean-shift.

        Parameters
        ----------
        cls : class with V_sel and mean attributes
        """
        self.U, self.T_k = cls.V_sel.copy(), cls.mean.copy()

        # CSR-format full blocks
        self.K_a = {
            region: {mat: _ensure_csr(block) for mat, block in mats.items()}
            for region, mats in self.K_list.items()
        }
        # project stiffness blocks and mean
        self.K_r_a = {
            region: {mat: (self.U.T @ B) @ self.U for mat, B in sub.items()}
            for region, sub in self.K_a.items()
        }
        self.K_a_Tk = {
            region: {mat: (self.U.T @ B) @ self.T_k for mat, B in sub.items()}
            for region, sub in self.K_a.items()
        }
        # project Neumann vector
        self.f_term = self.U.T @ self.rhs_linear

    def rom_solver(self, cls, param):
        """
        Solve the reduced-order model and reconstruct displacement.

        Parameters
        ----------
        param : (E, ν)

        Returns
        -------
        ms_full_sol : ndarray
            Reconstructed full-order displacement.
        T_mean_term : ndarray
            Mean-shift correction.
        """
        # restore FOM blocks
        self.K_list, self.rhs_linear = cls.K_list.item(), cls.rhs_linear

        if cls.cur_itr == 0:
            self.reduced_operators(cls)

        E, nu = param
        # fetch coefficients per region/material
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

        # assemble reduced stiffness and shift term
        self.K_r = sum(
            props[r][mat] * self.K_r_a[r][mat]
            for r in props for mat in props[r]
        )
        self.T_k_term = sum(
            props[r][mat] * self.K_a_Tk[r][mat]
            for r in props for mat in props[r]
        )

        # solve reduced system and reconstruct
        g             = self.f_term - self.T_k_term
        u_rom             = np.linalg.solve(self.K_r, g)

        return u_rom
    
    def hyper_rom_solver_deim(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass

    def hyper_rom_solver_ecsw(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass