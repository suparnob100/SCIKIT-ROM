"""
Problem 1: Piecewise 1-D heat conductivity (nonlinear)
"""
import numpy as np
import os
from skfem import asm
from skrom.fom.fem_utils import load_domain, newton_solver
from skrom.problem_classes.static.master_class import register_problem, Problem
from skrom.rom.linear_form_rom import LinearFormROM
from skrom.rom.rom_utils import newton_hyper_rom_solver, reconstruct_solution
from skrom.rom.ecsw.bilinear_form_hyperrom_ecsw import BilinearFormHYPERROM_ecsw
from skrom.rom.ecsw.linear_form_hyperrom_ecsw  import LinearFormHYPERROM_ecsw
from skfem import solve, condense



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
        from linear_forms import R
        return [R]

    def properties(self):
        """
        Return coefficient functions for each region.

        Returns
        -------
        list of callables
            [k, q] mapping parameters to conductivity/source.
        """
        from properties import deformation_gradient
        return [deformation_gradient]

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

    def assemble_kwargs(self, uv, param):
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
        return dict(
                    displacement = uv,
                    mu = param[0],
                    lmbda = param[1])

    def fom_operators(self, uv, param):
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
        displacement = uv
        kw = self.assemble_kwargs(displacement, param)
        J_mat = asm(self.a, self.basis, **kw)
        
        return J_mat
    
    def fom_rhs(self, uv, param):
        """
        Assemble FOM right-hand side.

        Parameters
        ----------
        uv : ndarray
            Temperature at DOFs.
        param : tuple
            (k_param, q_param).

        Returns
        -------
        R_vec : ndarray
            Full-order residual vector.
        """
        displacement = uv
        kw = self.assemble_kwargs(displacement, param)
        R_vec = asm(self.l, self.basis, **kw)

        return R_vec

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
            self.u0  = self.basis.zeros()
            du = self.basis.zeros()
            self.right = -0.1
            self.a, self.l = self.bilinear_forms()[0], self.linear_forms()[0]
        
        return self.newton_solver_fom(self.u0, param, nsteps = 7, tol=1e-3)
    
    def newton_solver_fom(self, u, param, nsteps, tol):
        for step in range(nsteps):
            for iteration in range(10):
                c = (step + 1.) / nsteps

                du_D = u.copy()
                x, y, z = self.basis.doflocs[:, self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^1']] = c * self.right - du_D[self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^2']] = (
                    y * np.cos(c * np.pi) - z * np.sin(c * np.pi) - y
                    - du_D[self.right_dofs.nodal['u^2']]
                )
                du_D[self.right_dofs.nodal['u^3']] = (
                    y * np.sin(c * np.pi) + z * np.cos(c * np.pi) - z
                    - du_D[self.right_dofs.nodal['u^3']]
                )

                uv = self.basis.interpolate(u)
                K, f = self.fom_operators(uv, param), self.fom_rhs(uv, param)
                
                du = solve(*condense(K, -f, x=du_D, D=self.dirichlet_dofs))
                norm_du = np.linalg.norm(du)
                u += du

                print(1 + iteration, norm_du)

                if norm_du < tol:
                    return u

    def reduced_operators(self, uv, param):
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
        J_mat, R_vec = self.fom_operators(uv, param), self.fom_rhs(uv, param)
        J_red = self.U.T @ (J_mat @ self.U)
        R_red = self.U.T @ R_vec
        return J_red, R_red
    
    def newton_solver_rom(self, u_rom, param, nsteps, tol):
        for step in range(nsteps):
            for iteration in range(10):
                c = (step + 1.) / nsteps
                u_full = reconstruct_solution(u_rom, self.U, self.T_k)

                du_D = u_full.copy()
                x, y, z = self.basis.doflocs[:, self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^1']] = c * self.right - du_D[self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^2']] = (
                    y * np.cos(c * np.pi) - z * np.sin(c * np.pi) - y
                    - du_D[self.right_dofs.nodal['u^2']]
                )
                du_D[self.right_dofs.nodal['u^3']] = (
                    y * np.sin(c * np.pi) + z * np.cos(c * np.pi) - z
                    - du_D[self.right_dofs.nodal['u^3']]
                )
                
                uv = self.basis.interpolate(u_full)
                K, f = self.reduced_operators(uv, param)
                
                du = np.linalg.solve(K, -f)
                norm_du = np.linalg.norm(du)
                u_rom += du

                print(1 + iteration, norm_du)

                if norm_du < tol:
                    return u_rom

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
            self.U, self.T_k, self.n_sel = cls.V_sel, cls.mean, cls.n_sel
            load_domain(self)
            self.u0_rom = np.full(self.n_sel, 0.0)
            du = self.basis.zeros()
            self.right = -0.1
            self.a, self.l = self.bilinear_forms()[0], self.linear_forms()[0]

        u_rom = self.newton_solver_rom(self.u0_rom, param, nsteps = 7, tol=1e-3)
        
        return u_rom

    def hyper_rom_operators(self, uv, param):
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
        mu, lmbda = param
        # reconstruct full-order field

        # assemble weighted hyper-reduced residual & Jacobian
        self.residual_hyper = self.residual_hyper_red.assemble_weighted_ecsw(
            displacement = uv, mu = mu, lmbda= lmbda
        )
        self.jac_hyper = self.jac_hyper_red.assemble_weighted_ecsw(
            displacement = uv,  mu = mu, lmbda= lmbda
        )

        return self.jac_hyper, self.residual_hyper
    
    def newton_solver_hyper_rom(self, u_rom, param, nsteps, tol):
        for step in range(nsteps):
            for iteration in range(10):
                c = (step + 1.) / nsteps
                u_full = reconstruct_solution(u_rom, self.U, self.T_k)

                du_D = u_full.copy()
                x, y, z = self.basis.doflocs[:, self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^1']] = c * self.right - du_D[self.right_dofs.nodal['u^1']]
                du_D[self.right_dofs.nodal['u^2']] = (
                    y * np.cos(c * np.pi) - z * np.sin(c * np.pi) - y
                    - du_D[self.right_dofs.nodal['u^2']]
                )
                du_D[self.right_dofs.nodal['u^3']] = (
                    y * np.sin(c * np.pi) + z * np.cos(c * np.pi) - z
                    - du_D[self.right_dofs.nodal['u^3']]
                )

                uv = self.basis.interpolate(u_full)
                K, f = self.hyper_rom_operators(uv, param)
                
                du = np.linalg.solve(K, -f)
                norm_du = np.linalg.norm(du)

                u_rom += du

                print(1 + iteration, norm_du)

                if norm_du < tol:
                    return u_rom


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
            self.U, self.T_k, self.n_sel, self.weights = cls.V_sel, cls.mean, cls.n_sel, cls.z

            load_domain(self)                                  # mesh, basis, DOFs
            self.R, self.J_form = self.linear_forms()[0], self.bilinear_forms()[0]

            # Hyper-reduced residual and Jacobian
            self.residual_hyper_red = LinearFormHYPERROM_ecsw(
                self.R, self.weights, self.basis, self.U, free_dofs=self.free_dofs
            )
            self.jac_hyper_red = BilinearFormHYPERROM_ecsw(
                self.J_form, self.weights, self.basis, self.U, self.U,
                free_dofs=self.free_dofs
            )

            # Elements kept by cubature
            self.elem_indices = np.nonzero(self.weights > 0)[0]
            self.u0 = np.full(self.n_sel, 0.0)               # cold start
            self.right = -0.1

        return self.newton_solver_hyper_rom(self.u0, param = param, nsteps = 7, tol = 1e-3)
    


    def hyper_rom_solver_deim(self):
        pass