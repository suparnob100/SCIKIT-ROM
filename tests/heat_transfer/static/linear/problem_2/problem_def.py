"""
Problem 2: Piecewise 1-D conductivity

Defines full-order and reduced-order workflows with separate affine
expansions for stiffness (K) and load (b) over two material regions.
"""
import numpy as np                           # array operations
from skfem import asm, condense, solve       # FEM assembly and solvers
from src.skrom.fom.fem_utils import load_domain
from src.skrom.rom.rom_utils import _ensure_csr
from src.skrom.problem_classes.master_class_static import register_problem, Problem

@register_problem("problem_2")
class ProblemAffine(Problem):
    """
    Affine reduced-order model for a two-region 1-D problem.

    Attributes
    ----------
    K_list : dict
        Full-order stiffness blocks per region.
    rhs_list : dict
        Full-order load vectors per region.
    """

    def domain(self):
        """
        Load domain geometry and FE basis data.
        Returns
        -------
        dict
            Mesh, basis, DOF mappings, and region partitioning.
        """
        from .domain import domain_
        return domain_()

    def bilinear_forms(self):
        """
        Return list of affine stiffness forms.
        Returns
        -------
        list of callable
            [a] where a(u, v) defines local stiffness.
        """
        from .bilinear_forms import a
        return [a]

    def linear_forms(self):
        """
        Return list of affine load forms.
        Returns
        -------
        list of callable
            [l] where l(v) defines local load.
        """
        from .linear_forms import l
        return [l]

    def properties(self):
        """
        Return property functions for stiffness and load.
        Returns
        -------
        list of callable
            [k, q] mapping parameters to local coefficient functions.
        """
        from .properties import k, q
        return [k, q]

    def parameters(self, n_samples):
        """
        Generate sampling of physical parameters.
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        Returns
        -------
        ndarray
            Sampled (k_param, q_param) pairs.
        """
        from .params import parameters
        return parameters(n_samples)

    def fom_operators(self, cls):
        """
        Assemble full-order stiffness and load by region.
        On first iteration, caches region-wise blocks in cls.K_list and cls.rhs_list.
        """
        if cls.cur_itr == 0:
            load_domain(self)  # initialize mesh, basis, DOFs, regions
            a, l = self.bilinear_forms()[0], self.linear_forms()[0]

            # assemble per-region stiffness and load
            cls.K_list   = {r: asm(a, b) for r, b in self.basis_regions.items()}
            cls.rhs_list = {r: asm(l, b) for r, b in self.basis_regions.items()}

        return cls.K_list, cls.rhs_list

    def fom_solver(self, cls, param):
        """
        Solve full-order system with Dirichlet BCs.
        Scales each region block by its parameter-dependent coefficient.
        """
        self.K_list, self.rhs_list = self.fom_operators(cls)

        # global stiffness = sum_k [ k(param) * K_region ]
        K = sum(
            self.properties()[0](param[0], region) * Kmat
            for region, Kmat in self.K_list.items()
        )
        # global load = sum_q [ q(param) * load_region ]
        rhs = sum(
            self.properties()[1](param[1], region) * bmat
            for region, bmat in self.rhs_list.items()
        )

        # apply Dirichlet BCs and solve reduced system
        u = self.basis.zeros()
        u[self.dirichlet_boundary_dofs] = self.dirichlet_boundary_value
        return solve(*condense(K, rhs, x=u, I=self.free_dofs))

    def reduced_operators(self, cls):
        """
        Build reduced stiffness and load projection terms.
        Projects each region block onto the reduced basis U.
        """
        self.U, self.T_k = cls.V_sel, cls.mean

        # ensure correct sparse format
        Kc_dict = {region: _ensure_csr(Kblock) for region, Kblock in self.K_list.items()}

        # project stiffness and load per region
        self.K_r_a = {
            region: self.U.T @ (Kc @ self.U)
            for region, Kc in Kc_dict.items()
        }
        self.q_term = {
            region: self.U.T @ self.rhs_list[region]
            for region in Kc_dict
        }
        self.k_term = {
            region: self.U.T @ (Kc @ self.T_k)
            for region, Kc in Kc_dict.items()
        }

    def rom_solver(self, cls, param):
        """
        Solve reduced-order model and reconstruct solution.
        Assembles reduced stiffness and residual, solves for modal coefficients.
        """
        # retrieve cached full-order blocks
        self.K_list, self.rhs_list = cls.K_list.item(), cls.rhs_list.item()
        k_param, q_param = param

        if cls.cur_itr == 0:
            # compute reduced operators on first call
            self.reduced_operators(cls)

        # compute per-region coefficients
        kf, qf = self.properties()
        coefs = {
            region: {'k': kf(k_param, region), 'q': qf(q_param, region)}
            for region in self.K_list
        }

        # assemble reduced stiffness and RHS vector
        self.K_r = sum(coefs[r]['k'] * self.K_r_a[r] for r in coefs)
        g       = sum(
            coefs[r]['q'] * self.q_term[r] - coefs[r]['k'] * self.k_term[r]
            for r in coefs
        )

        # solve for modal coefficients and reconstruct full solution
        A = np.linalg.solve(self.K_r, g)
        return self.U @ A, cls.mean