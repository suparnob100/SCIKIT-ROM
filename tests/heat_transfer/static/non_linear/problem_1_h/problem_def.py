"""
Problem 1: Piecewise 1-D heat conductivity (nonlinear)
"""
import numpy as np
from skfem import asm
from src.skrom.fom.fem_utils import load_domain, newton_solver
from src.skrom.rom.rom_utils import newton_solver_rom
from src.skrom.problem_classes.master_class_static import register_problem, Problem
from src.skrom.rom.linear_form_rom import LinearFormROM
from src.skrom.rom.rom_utils import newton_hyper_rom_solver, reconstruct_solution
from src.skrom.rom.bilinear_form_hyperrom import BilinearFormHYPERROM
from src.skrom.rom.linear_form_hyperrom  import LinearFormHYPERROM


@register_problem("problem_1_h")
class ProblemNonLinear(Problem):
    """
    Affine ROM for a two-region 1-D heat conduction problem.

    Attributes
    ----------
    K_list : dict
        Full-order stiffness matrices by region.
    rhs_list : dict
        Full-order load vectors by region.
    """

    def domain(self):
        """
        Load mesh, basis, DOFs, and region partitioning.

        Returns
        -------
        dict
            Domain data for FOM assembly.
        """
        from .domain import domain_
        return domain_()

    def bilinear_forms(self):
        """
        Return affine stiffness form(s).

        Returns
        -------
        list of callables
            [J_form] for local stiffness assembly.
        """
        from .bilinear_forms import J_form
        return [J_form]

    def linear_forms(self):
        """
        Return affine load form(s).

        Returns
        -------
        list of callables
            [R] for local load assembly.
        """
        from .linear_forms import R
        return [R]

    def properties(self):
        """
        Return coefficient functions for each region.

        Returns
        -------
        list of callables
            [k, q] mapping parameters to conductivity/source.
        """
        from .properties import k, q
        return [k, q]

    def parameters(self, n_samples):
        """
        Generate samples of physical parameters.

        Parameters
        ----------
        n_samples : int
            Number of (k_param, q_param) samples.

        Returns
        -------
        ndarray, shape (n_samples, 2)
            Sampled parameter tuples.
        """
        from .params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, global_mask, u, param):
        """
        Pack kwargs for assembler.

        Parameters
        ----------
        global_mask : dict
            Region masks.
        u : ndarray
            Current temperature field.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        dict
            {'global_mask', 'u_prev', 'k_param', 'q_param'}.
        """
        return dict(global_mask=global_mask,
                    u_prev=u,
                    k_param=param[0],
                    q_param=param[1])

    def fom_operators(self, u, param):
        """
        Assemble FOM Jacobian and residual.

        Parameters
        ----------
        u : ndarray
            Temperature at DOFs.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        J_mat : csr_matrix
            Full-order Jacobian.
        R_vec : ndarray
            Full-order residual vector.
        """
        u_prev = self.basis.interpolate(u)
        kw = self.assemble_kwargs(self.global_mask, u_prev, param)
        R_vec, J_mat = asm(self.l, self.basis, **kw), asm(self.a, self.basis, **kw)
        
        return J_mat, R_vec

    def fom_solver(self, cls, param):
        """
        Solve FOM via Newton’s method.

        Parameters
        ----------
        cls : class
            Provides mesh/boundary data on first iteration.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        u_sol : ndarray
            Converged temperature field.
        """
        if cls.cur_itr == 0:
            load_domain(self)
            self.u0 = self.basis.zeros() + 273.0
            self.D = self.dirichlet_boundary_dofs.nodal_ix
            self.a, self.l = self.bilinear_forms()[0], self.linear_forms()[0]

        return newton_solver(
            self.fom_operators,
            self.u0,
            self.D,
            self.dirichlet_boundary_value,
            param,
            tol=1e-2,
            maxit=50
        )

    def reduced_operators(self, u, param):
        """
        Project FOM operators onto reduced basis.

        Parameters
        ----------
        u : ndarray
            Reduced coefficients or previous solution.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        J_red : ndarray
            Reduced Jacobian.
        R_red : ndarray
            Reduced residual.
        """
        J_mat, R_vec = self.fom_operators(u, param)
        J_red = self.U.T @ (J_mat @ self.U)
        R_red = self.U.T @ R_vec
        return J_red, R_red

    def rom_solver(self, cls, param):
        """
        Solve ROM via Newton’s method and reconstruct solution.

        Parameters
        ----------
        cls : class
            Provides reduced basis and mean on first iteration.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        u_full : ndarray
            Reconstructed full-order solution.
        T_mean : ndarray
            Mean term added to reduced solution.
        """
        if cls.cur_itr == 0:
            # initialize ROM state
            self.T_k, self.U, self.n_sel = cls.mean, cls.V_sel, cls.n_sel
            load_domain(self)
            self.u0_rom = np.full(self.n_sel, 500.0)
            self.D = self.dirichlet_boundary_dofs.nodal_ix
            self.a, self.l = self.bilinear_forms()[0], self.linear_forms()[0]

        return newton_solver_rom(
            self.reduced_operators,
            self.u0_rom,
            param,
            V_sel=self.U,
            mean=self.T_k,
            tol=1e-2,
            maxit=50
        )

    def hyper_rom_operators(self, u, param):
        """
        Compute hyper-reduced Jacobian and residual for given ROM state.

        Parameters
        ----------
        u : ndarray
            Reduced coefficients.
        param : tuple
            (k_param, q_param) physical parameters.

        Returns
        -------
        jac_hyper : sparse matrix
            Hyper-reduced Jacobian.
        residual_hyper : ndarray
            Hyper-reduced residual.
        """
        k_param, q_param = param
        # reconstruct full-order field
        u_full = reconstruct_solution(u, self.U, self.T_k)

        # assemble weighted hyper-reduced residual & Jacobian
        self.residual_hyper = self.residual_hyper_red.assemble_weighted(
            u_prev=u_full, k_param=k_param, q_param=q_param,
            global_mask=self.global_mask
        )
        self.jac_hyper = self.jac_hyper_red.assemble_weighted(
            u_prev=u_full, k_param=k_param, q_param=q_param,
            global_mask=self.global_mask
        )

        return self.jac_hyper, self.residual_hyper

    def hyper_rom_solver(self, cls, param):
        """
        Assemble hyper-reduced operators (first call) and run Newton solver.

        Parameters
        ----------
        cls : object
            ROM controller carrying basis, mean, and weights.
        param : tuple
            Physical parameters passed to `hyper_rom_operators`.

        Returns
        -------
        ndarray
            Converged reduced coefficients.
        """
        if cls.cur_itr == 0:                                   # first snapshot → build once
            # ROM data & weights
            self.U, self.T_k, self.n_sel, self.weights = cls.V_sel, cls.mean, cls.n_sel, cls.z

            load_domain(self)                                  # mesh, basis, DOFs
            self.R, self.J_form = self.linear_forms()[0], self.bilinear_forms()[0]

            # Hyper-reduced residual and Jacobian
            self.residual_hyper_red = LinearFormHYPERROM(
                self.R, self.weights, self.basis, self.U, free_dofs=self.free_dofs
            )
            self.jac_hyper_red = BilinearFormHYPERROM(
                self.J_form, self.weights, self.basis, self.U, self.U,
                free_dofs=self.free_dofs
            )

            # Elements kept by cubature
            self.elem_indices = np.nonzero(self.weights > 0)[0]
            self.u0 = np.full(self.n_sel, 273.0)               # cold start

        return newton_hyper_rom_solver(self.hyper_rom_operators, self.u0, param = param)