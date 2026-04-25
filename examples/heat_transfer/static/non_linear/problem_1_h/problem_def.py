"""
Problem 1: Piecewise 1-D heat conductivity (nonlinear)
"""
import numpy as np
import os
from skfem import asm
from skrom.fom.fem_utils import load_domain, newton_solver
from skrom.rom.rom_utils import newton_solver_rom
from skrom.problem_classes.masterclass import register_problem, Problem
from skrom.rom.linear_form_rom import LinearFormROM
from skrom.rom.rom_utils import newton_hyper_rom_solver, reconstruct_solution
from skrom.rom.ecsw.bilinear_form_hyperrom_ecsw import BilinearFormHYPERROM_ecsw
from skrom.rom.ecsw.linear_form_hyperrom_ecsw  import LinearFormHYPERROM_ecsw
from skrom.rom.ecm.bilinear_form_hyperrom_ecm import BilinearFormHYPERROM_ecm
from skrom.rom.ecm.linear_form_hyperrom_ecm  import LinearFormHYPERROM_ecm
from skrom.rom.deim.bilinear_form_hyperrom_deim import BilinearFormHYPERROM_deim
from skrom.rom.deim.linear_form_hyperrom_deim  import LinearFormHYPERROM_deim

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))
# global_mask = None

@register_problem(PROBLEM_NAME)
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
        from domain import domain_
                    
        return domain_()

    def bilinear_forms(self):
        """
        Return affine stiffness form(s).

        Returns
        -------
        list of callables
            [J_form] for local stiffness assembly.
        """
        from bilinear_forms import J_form
        return [J_form]

    def linear_forms(self):
        """
        Return affine load form(s).

        Returns
        -------
        list of callables
            [R] for local load assembly.
        """
        from linear_forms import R, rhs
        return [R, rhs]

    def properties(self):
        """
        Return coefficient functions for each region.

        Returns
        -------
        list of callables
            [k, q] mapping parameters to conductivity/source.
        """
        from properties import k, q
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
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, u, param):
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
        # """
        # return dict(global_mask=global_mask,
        #             u_prev=u,
        #             k_param=param[0],
        #             q_param=param[1])
        
        return dict(u_prev=u,
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
        # kw = self.assemble_kwargs(self.global_mask, u_prev, param)
        kw = self.assemble_kwargs(u_prev, param)
        kw.update({ 'global_mask': self.global_mask})
    
        J_mat = asm(self.a, self.basis, **kw)
        
        return J_mat
    
    def fom_rhs(self, u, param):
        u_prev = self.basis.interpolate(u)
        kw = self.assemble_kwargs(u_prev, param)
        kw.update({ 'global_mask': self.global_mask})
        R_vec =  asm(self.l, self.basis, **kw)

        return R_vec
    
    def f_nl(self, u, param):
        u_prev = self.basis.interpolate(u)
        kw = self.assemble_kwargs(u_prev, param)
        kw.update({ 'global_mask': self.global_mask})
        rhs = asm(self.linear_forms()[1], self.basis, **kw)

        return rhs

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
            self.fom_rhs,
            self.u0,
            self.D,
            self.dirichlet_boundary_value,
            param,
            tol=1e-2,
            maxit=50,
            rhs_args=(param,)
        )

            

    

    def reduced_operators(self, u_rom, param):
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
        u_full = reconstruct_solution(u_rom, self.U, self.T_ref)
        J_mat, R_vec = self.fom_operators(u_full, param), self.fom_rhs(u_full, param)
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
            self.T_ref, self.U, self.n_sel = cls.train_ref, cls.V_sel, cls.n_sel
            load_domain(self)
            
            self.u0_rom = np.full(self.n_sel, 500.0)
            self.D = self.dirichlet_boundary_dofs.nodal_ix
            self.a, self.l = self.bilinear_forms()[0], self.linear_forms()[0]

        return newton_solver_rom(
            self.reduced_operators,
            self.u0_rom,
            param,
            tol=1e-2,
            maxit=50
        )
    
    def hyper_rom_operators_ecsw(self, u, param):
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
        # reconstruct full-order field
        u_full = reconstruct_solution(u, self.U, self.T_ref)

        kw = self.assemble_kwargs(u_full, param)
        kw.update({ 'global_mask': self.global_mask})

        # assemble weighted hyper-reduced residual & Jacobian
        self.residual_hyper = self.residual_hyper_red_ecsw.assemble_weighted_ecsw(**kw)
        self.jac_hyper = self.jac_hyper_red_ecsw.assemble_weighted_ecsw(**kw)

        return self.jac_hyper, self.residual_hyper

    def hyper_rom_solver_ecsw(self, cls, param):
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
            self.U, self.T_ref, self.n_sel, self.weights = cls.V_sel, cls.train_ref, cls.n_sel, cls.z

            load_domain(self)                                  # mesh, basis, DOFs
            self.R, self.J_form = self.linear_forms()[0], self.bilinear_forms()[0]

            # Hyper-reduced residual and Jacobian
            self.residual_hyper_red_ecsw = LinearFormHYPERROM_ecsw(
                self.R, self.weights, self.basis, self.U, free_dofs=self.free_dofs
            )
            self.jac_hyper_red_ecsw = BilinearFormHYPERROM_ecsw(
                self.J_form, self.weights, self.basis, self.U, self.U,
                free_dofs=self.free_dofs
            )

            # Elements kept by cubature
            self.elem_indices = np.nonzero(self.weights > 0)[0]
            self.u0 = np.full(self.n_sel, 273.0)               # cold start

        return newton_solver_rom(self.hyper_rom_operators_ecsw, self.u0, param = param, tol = 1e-2, maxit = 50)
    

    def hyper_rom_operators_ecm(self, u, param):
        """
        Compute ECM hyper-reduced Jacobian and residual for given ROM state.
        """
        u_full = reconstruct_solution(u, self.U, self.T_ref)

        kw = self.assemble_kwargs(u_full, param)
        kw.update({'global_mask': self.global_mask})

        self.residual_hyper = self.residual_hyper_red_ecm.assemble_weighted_ecm(**kw)
        self.jac_hyper = self.jac_hyper_red_ecm.assemble_weighted_ecm(**kw)

        return self.jac_hyper, self.residual_hyper

    def hyper_rom_solver_ecm(self, cls, param):
        """
        Assemble ECM hyper-reduced operators on the first call and run Newton.
        """
        if cls.cur_itr == 0:
            self.U, self.T_ref, self.n_sel, self.gauss_weights = cls.V_sel, cls.train_ref, cls.n_sel, cls.z

            load_domain(self)
            self.R, self.J_form = self.linear_forms()[0], self.bilinear_forms()[0]

            self.residual_hyper_red_ecm = LinearFormHYPERROM_ecm(
                self.R, self.gauss_weights, self.basis, self.U, free_dofs=self.free_dofs
            )
            self.jac_hyper_red_ecm = BilinearFormHYPERROM_ecm(
                self.J_form, self.gauss_weights, self.basis, self.U, self.U,
                free_dofs=self.free_dofs
            )

            self.elem_indices = np.nonzero(np.any(np.asarray(self.gauss_weights) > 0, axis=1))[0]
            self.u0 = np.full(self.n_sel, 273.0)

        return newton_solver_rom(self.hyper_rom_operators_ecm, self.u0, param=param, tol=1e-2, maxit=50)

    def hyper_rom_operators_deim(self, u, param):
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
        # reconstruct full-order field
        u_full = reconstruct_solution(u, self.U, self.T_ref)

        kw = self.assemble_kwargs(u_full, param)
        kw.update({ 'global_mask': self.global_mask})

        # assemble weighted hyper-reduced residual & Jacobian
        self.residual_hyper = self.residual_hyper_red_deim.assemble_deim(**kw)
        self.jac_hyper = self.jac_hyper_red_deim.assemble_deim(**kw)

        return self.jac_hyper, self.residual_hyper

    def hyper_rom_solver_deim(self, cls, param):
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
            self.U, self.T_ref, self.n_sel, self.weights = cls.V_sel, cls.train_ref, cls.n_sel, cls.z
            self.sampled_rows = cls.sampled_rows
            self.deim_mat = cls.deim_mat

            load_domain(self)                                  # mesh, basis, DOFs
            self.R, self.J_form = self.linear_forms()[0], self.bilinear_forms()[0]

            # Hyper-reduced residual and Jacobian
            self.residual_hyper_red_deim = LinearFormHYPERROM_deim(
                self.R, self.weights, self.basis, self.U, sampled_rows = self.sampled_rows, deim_mat = self.deim_mat, free_dofs=self.free_dofs
            )
            self.jac_hyper_red_deim = BilinearFormHYPERROM_deim(
                self.J_form, self.weights, self.basis, self.U, self.U, sampled_rows = self.sampled_rows, deim_mat = self.deim_mat, 
                free_dofs=self.free_dofs
            )

            # Elements kept by cubature
            self.elem_indices = np.nonzero(self.weights > 0)[0]
            self.u0 = np.full(self.n_sel, 273.0)               # cold start

        return newton_solver_rom(self.hyper_rom_operators_deim, self.u0, param = param, tol = 0.01, maxit = 50,alpha=1.0)
