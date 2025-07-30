"""
Problem_1: Linear Elasticity in a static setting (Affine)
"""
from src.skrom.problem_classes.master_class_static import register_problem, Problem
import numpy as np
from src.skrom.rom.rom_utils import _ensure_csr
from src.skrom.fom.fem_utils import load_domain
from src.skrom.utils.dynamics.integrators import newmark_with_damping
from skfem import asm
from scipy.sparse import csc_matrix

@register_problem("problem_1")
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
        from .domain import domain_
        return domain_()


    def bilinear_forms(self):
        """
        Collect affine bilinear forms for λ and μ.

        Returns
        -------
        [stiffness_lam, stiffness_mu]
        """
        from .bilinear_forms import stiffness_lam, stiffness_mu, mass
        return [stiffness_lam, stiffness_mu, mass]


    def linear_forms(self):
        """
        Collect affine linear form for traction.

        Returns
        -------
        [traction]
        """
        from .linear_forms import traction
        return [traction]


    def properties(self):
        """
        Return function mapping (E, ν, region) → (λ, μ).

        Returns
        -------
        callable
            lame_params(E, ν, region)
        """
        from .properties import lame_params
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
        from .params import parameters
        return parameters(n_samples)


    def assemble_kwargs(self):
        """
        Pack physical parameters for assembly routines.

        Parameters
        ----------
        param : tuple of float
            (E, nu)

        Returns
        -------
        dict
            {'E': E, 'nu': nu}
        """

        t = getattr(self, 'time', None)

        # return dict(E=param[0], nu=param[1], time = t)
        return dict(time = t)


    def time_vector(self):
        return np.linspace(0,8000,500*5)


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

            mass_form = self.bilinear_forms()[2]
            cls.M = asm(mass_form, self.basis)

        return cls.M, cls.K_list


    def fom_rhs(self, cls):
        
        if cls.cur_itr == 0:

            b_linear = self.linear_forms()[0]
            # assemble Neumann load once
            kw  = self.assemble_kwargs()
            cls.rhs_linear = asm(b_linear, self.fbasis_neumann, **kw)

        return cls.rhs_linear


    def fom_solver(self, cls, param):
        """
        Full-order 3D linear elasticity with Guyan (static) condensation
        of stiffness and mass, and efficient RHS condensation.
        """

        # 1) Time setup
        times = self.time_vector()
        dt    = times[1] - times[0]
        self.time = times[0]
        n_steps   = len(times)

        # 2) Assemble full-order operators
        M0, K_blocks = self.fom_operators(cls)
        E, nu = param

        # Build full stiffness matrix K0
        K0 = sum(
            block * coef
            for region, mats in K_blocks.items()
            for block, coef in zip(
                mats.values(),
                self.properties()(E, nu, region)
            )
        ).tocsr()
        M0 = M0.tocsr()

        # 3) Dirichlet BC vector (static)
        u_bc_full = self.basis.zeros()
        u_bc_full[self.dirichlet_dofs] = self.dirichlet_boundary_value

        # 4) Free vs removed DOFs
        all_dofs = np.arange(K0.shape[0])
        remove  = self.dirichlet_dofs
        free    = np.setdiff1d(all_dofs, remove, assume_unique=True)


        from scipy.sparse.linalg import eigsh
        # 1) convert to CSC for efficient factorization / eigen‐solve

        def sub(A, rows, cols):
            return csc_matrix(A[rows, :][:, cols])

        # 7) extract free‐free blocks
        M_ff = M0[free][:, free].tocsr()
        K_ff = K0[free][:, free].tocsr()

        # 2) choose Rayleigh-damping as pure mass-prop:
        cv = 1e-3
        cm  = 1e-3
            
        C_ff = cv * M_ff.copy() + cm * K_ff.copy()

        # 8) extract free‐remove coupling blocks
        M_fr = M0[free][:, remove].toarray()
        K_fr = K0[free][:, remove].toarray()
        C_fr = cv * M_fr.copy() + cm * K_fr.copy()


        # 9) RHS‐condensation helper
        def condense_rhs_dynamic(F_full, uD=np.zeros(len(remove)), uD_dot=np.zeros(len(remove)), uD_ddot=np.zeros(len(remove))):
            f = F_full.ravel()
            return ( f[free]
                - M_fr @ uD_ddot
                - C_fr @ uD_dot
                - K_fr @ uD )


        # 11) Newmark callback for reduced load
        if cls.cur_itr == 0:
            self.F_free_mat = []


        def force_free(i, times):
            if cls.cur_itr == 0:
                self.time = times[i]
                F_full = self.fom_rhs(cls)

                # known Dirichlet trace and its time‐derivatives
                u_Dirichlet      = u_bc_full[remove]
                # xD_dot  = self.bc_velocity(times[i])
                # xD_ddot = self.bc_acceleration(times[i])

                self.F_free_mat.append(
                    condense_rhs_dynamic(F_full, uD=u_Dirichlet)
                )
            return self.F_free_mat[i]
        

        # 7) Initial reduced DOFs
        U0_c = u_bc_full[free].copy()
        V0_c = np.zeros_like(U0_c)


        # 9) Time-integrate on reduced system
        U_c, V_c, A_c = newmark_with_damping(
            M_ff, C_ff, K_ff,
            force_free,
            times,
            U0=U0_c, V0=V0_c,
        )

        # 10) Rebuild full-field histories
        n_dof    = K0.shape[0]
        U_full   = np.zeros((n_dof, n_steps))
        V_full   = np.zeros_like(U_full)
        A_full   = np.zeros_like(U_full)


        U_full[free, :]                = U_c
        U_full[self.dirichlet_dofs, :] = self.dirichlet_boundary_value
        V_full[free, :] = V_c
        A_full[free, :] = A_c


        return U_full, V_full, A_full


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
        A             = np.linalg.solve(self.K_r, g)
        ms_full_sol   = self.U @ A
        T_mean_term   = cls.mean
        return ms_full_sol, T_mean_term
    

    def hyper_rom_solver(self):
        """Solve hyper-reduced-order model for given parameters."""
        pass