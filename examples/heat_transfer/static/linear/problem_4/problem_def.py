"""
Problem_4: Heat conduction on a star-shaped domain (Affine)
"""
import numpy as np
import os
from skfem import asm, condense, solve
from skrom.fom.fem_utils import load_domain
from skrom.rom.rom_utils import _ensure_csr
from skrom.problem_classes.static.master_class import register_problem, Problem

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))

@register_problem(PROBLEM_NAME)
class ProblemMixed(Problem):
    """
    Defines FOM and ROM workflows for problem_4.

    Attributes
    ----------
    K_bilinear : sparse matrix
        Full-order stiffness matrix.
    rhs_linear : ndarray
        Full-order load vector.
    """

    def domain(self):
        """
        Load domain configuration.

        Returns
        -------
        dict
            Mesh, basis, DOFs, and region partitioning.
        """
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        """
        Return affine bilinear form for stiffness.
        """
        from bilinear_forms import a
        return [a]

    def linear_forms(self):
        """
        Return affine linear form for load.
        """
        from linear_forms import l
        return [l]

    def properties(self):
        """
        Return list of property functions [q()].
        """
        from properties import q
        return [q]

    def parameters(self, n_samples):
        """
        Generate snapshot parameters.

        Returns
        -------
        list of tuple
            [(q1,), (q2,), ...]
        """
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, param):
        """
        Pack parameters for assembly.

        Returns
        -------
        dict
            {'q_param': float}
        """
        return dict(q_param=param[0])

    def fom_operators(self, cls):
        """
        Assemble or reuse full-order operators.

        Parameters
        ----------
        param : tuple
            (q_param,)

        Returns
        -------
        K_bilinear, rhs_linear
        """

        if cls.cur_itr == 0:
            # assemble stiffness and load domain once
            a = self.bilinear_forms()[0]
            load_domain(self)
            if not hasattr(cls, "K_bilinear"):
                cls.K_bilinear = asm(a, self.basis)

        return cls.K_bilinear
    
    def fom_rhs(self, cls, param):
        l = self.linear_forms()[0]
        kw = self.assemble_kwargs(param)
        cls.rhs_linear = asm(l, self.basis, **kw)

        return cls.rhs_linear


    def fom_solver(self, cls, param):
        """
        Solve FOM with Dirichlet BCs.

        Returns
        -------
        u_sol : ndarray
        """
        self.K_bilinear, self.rhs_linear = self.fom_operators(cls), self.fom_rhs(cls, param)

        u = self.basis.zeros()
        u[self.dirichlet_boundary_dofs] = self.dirichlet_boundary_value

        # solve condensed system
        return solve(*condense(self.K_bilinear, self.rhs_linear,
                               x=u, I=self.free_dofs))

    def reduced_operators(self, cls):
        """
        Compute reduced stiffness and mean-shift term.
        """
        # fetch reduced basis and mean
        self.U, self.T_k = cls.V_sel.copy(), cls.train_ref.copy()
        # project full-order stiffness
        K_a = _ensure_csr(self.K_bilinear)
        self.K_r = self.U.T @ (K_a @ self.U)
        # compute mean-shift
        temp = K_a @ self.T_k
        self.A_mat = np.linalg.solve(self.K_r, self.U.T @ temp)

    def rom_solver(self, cls, param):
        """
        Solve ROM and reconstruct solution.

        Returns
        -------
        ms_full_sol : ndarray
        T_mean_term : ndarray
        """
        # assemble full-order for this parameter
        self.K_bilinear, self.b_full = self.fom_operators(cls), self.fom_rhs(cls, param)
        f_a = np.asarray(self.b_full, float).ravel()

        if cls.cur_itr == 0:
            self.reduced_operators(cls)
        # solve reduced system
        u_rom = np.linalg.solve(self.K_r, self.U.T @ f_a) - self.A_mat

        return u_rom
    
    def hyper_rom_solver_ecsw(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass

    def hyper_rom_solver_deim(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass