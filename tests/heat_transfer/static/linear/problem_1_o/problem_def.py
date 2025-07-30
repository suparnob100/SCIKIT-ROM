import numpy as np
from skfem import asm, condense, solve
from src.skrom.fom.fem_utils import load_domain
from src.skrom.rom.rom_utils import _ensure_csr
from src.skrom.problem_classes.master_class_static import register_problem, Problem


@register_problem("problem_1")
class ProblemAffine(Problem):
    """
    Affine ROM for a linear-elastic problem with separate parameter-dependent expansions.
    """

    def domain(self):
        """Load mesh, basis, DOFs, and boundary regions from domain.py."""
        from .domain import domain_
        return domain_()

    def bilinear_forms(self):
        """
        Return affine bilinear form(s) for stiffness assembly.

        Returns
        -------
        list of callables
            Each callable a(u, v, p) contributes to the stiffness matrix.
        """
        from .bilinear_forms import a
        return [a]

    def linear_forms(self):
        """
        Return affine linear form(s) for load assembly.

        Returns
        -------
        list of callables
            Each callable l(v, p) contributes to the load vector.
        """
        from .linear_forms import l
        return [l]

    def properties(self):
        """
        Return functions mapping physical parameters to affine coefficients.

        Returns
        -------
        [k, q]
            k(param_k) scales stiffness; q(param_q) scales load.
        """
        from .properties import k, q
        return [k, q]

    def parameters(self, n_samples):
        """
        Generate sampling design over parameter space.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        ndarray
            Array of shape (n_samples, 2) with (k_param, q_param) pairs.
        """
        from .params import parameters
        return parameters(n_samples)

    def fom_operators(self, cls):
        """
        Assemble and cache full-order stiffness and load operators.

        On first call (cls.cur_itr == 0), build and store:
          - cls.K   : stiffness matrix (CSR)
          - cls.rhs : load vector

        Returns
        -------
        cls.K, cls.rhs
        """
        if cls.cur_itr == 0:
            load_domain(self)
            a  = self.bilinear_forms()[0]
            cls.K = asm(a, self.basis)

        return cls.K

    def fom_rhs(self, cls):
        if cls.cur_itr == 0:
            l = self.linear_forms()[0]
            cls.rhs = asm(l,self.basis)
        
        return cls.rhs
        
    def fom_solver(self, cls, param):
        """
        Solve the full-order system for given parameters.

        Parameters
        ----------
        cls : object
            Holds cached FOM operators and iteration state.
        param : tuple
            (k_param, q_param) physical parameters.

        Returns
        -------
        u_sol : ndarray
            Solution vector satisfying Dirichlet BCs.
        """
        # get cached operators
        K_bilinear, rhs_linear = self.fom_operators(cls), self.fom_rhs(cls)

        # scale by parameter-dependent properties
        K, rhs = (self.properties()[0](param[0]) * K_bilinear,
          self.properties()[1](param[1]) * rhs_linear)

        # apply Dirichlet BCs
        u = self.basis.zeros()
        u[self.dirichlet_boundary_dofs] = self.dirichlet_boundary_value

        # condense and solve
        u_sol = solve(
            *condense(K, rhs, x=u, I=self.free_dofs)
        )
        return u_sol

    def reduced_operators(self, cls, param):
        """
        Project FOM operators onto the reduced basis and compute mean term.

        Parameters
        ----------
        cls : object
            Contains V_sel (basis) and mean field.
        param : tuple
            Not used directly; projection is parameter-independent here.

        Returns
        -------
        ms_sol_full : ndarray
            Unscaled modal solution in full space.
        T_mean_term : ndarray
            Correction term from mean field.
        """
        self.U, self.T_k  = cls.V_sel, cls.mean

        # ensure CSR format for matrix operations
        K_a = _ensure_csr(self.K_full)
        f_a = np.asarray(self.b_full, float).ravel()

        # reduced stiffness
        K_r = self.U.T @ (K_a @ self.U)
        # compute modal coefficients and back-project
        a_vec       = np.linalg.solve(K_r, self.U.T @ f_a)
        self.ms_sol_full = self.U @ a_vec

        # compute mean-term correction
        temp          = K_a @ cls.mean
        A_mat         = np.linalg.solve(K_r, self.U.T @ temp)
        self.T_mean_term = cls.mean - (self.U @ A_mat)

        return self.ms_sol_full, self.T_mean_term

    def rom_solver(self, cls, param):
        """
        Solve the reduced-order model and reconstruct the solution.

        Parameters
        ----------
        cls : object
            Provides cached FOM operators, basis V_sel, and mean.
        param : tuple
            (k_param, q_param) for final scaling of modal solution.

        Returns
        -------
        ms_sol_param : ndarray
            Parameter-scaled modal solution in full space.
        T_mean_term : ndarray
            Mean correction term.
        """
        # cache full-order operators for ROM projection
        self.K_full, self.b_full = cls.K.item().copy(), cls.rhs.copy()

        if cls.cur_itr == 0:
            self.ms_sol_full, self.T_mean_term = self.reduced_operators(cls, param)

        # scale modal solution by property ratios
        ms_sol_param = (
            self.properties()[1](param[1])
            / self.properties()[0](param[0])
        ) * self.ms_sol_full

        return ms_sol_param, self.T_mean_term
    
    def hyper_rom_solver(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass