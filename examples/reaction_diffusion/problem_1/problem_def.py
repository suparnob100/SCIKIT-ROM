"""
Three-material diffusion-reaction problem with piecewise-constant
D, Sigma, and Q in each material zone.

Parameter vector:
    [D1, D2, D3, Sigma1, Sigma2, Sigma3, Q1, Q2, Q3]
"""
import numpy as np
import os
from skfem import asm, condense, solve
from skrom.fom.fem_utils import load_domain
from skrom.rom.rom_utils import _ensure_csr
from skrom.problem_classes.masterclass import register_problem, Problem

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))

@register_problem(PROBLEM_NAME)
class ProblemDiffusionReaction3Mat(Problem):

    def domain(self):
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        from bilinear_forms import a_diff, a_reac
        return [a_diff, a_reac]

    def linear_forms(self):
        from linear_forms import l_source
        return [l_source]

    def properties(self):
        return []

    def parameters(self, n_samples):
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs(self, param):
        return {
            "D1": param[0],
            "D2": param[1],
            "D3": param[2],
            "Sigma1": param[3],
            "Sigma2": param[4],
            "Sigma3": param[5],
            "Q1": param[6],
            "Q2": param[7],
            "Q3": param[8],
        }

    def fom_operators(self, cls, param):
        """
        Assemble/reuse affine pieces.

        Full-order operator:
            A = D1*K1 + D2*K2 + D3*K3 + Sigma1*M1 + Sigma2*M2 + Sigma3*M3

        Full-order RHS:
            b = Q1*f1 + Q2*f2 + Q3*f3
        """
        D1, D2, D3, Sigma1, Sigma2, Sigma3, Q1, Q2, Q3 = param

        if cls.cur_itr == 0 or not hasattr(cls, "K1"):
            load_domain(self)

            a_diff, a_reac = self.bilinear_forms()
            l_source = self.linear_forms()[0]

            cls.K1 = asm(a_diff, self.basis_mat1)
            cls.K2 = asm(a_diff, self.basis_mat2)
            cls.K3 = asm(a_diff, self.basis_mat3)

            cls.M1 = asm(a_reac, self.basis_mat1)
            cls.M2 = asm(a_reac, self.basis_mat2)
            cls.M3 = asm(a_reac, self.basis_mat3)

            cls.f1 = asm(l_source, self.basis_mat1)
            cls.f2 = asm(l_source, self.basis_mat2)
            cls.f3 = asm(l_source, self.basis_mat3)

        A = (
            D1 * cls.K1
            + D2 * cls.K2
            + D3 * cls.K3
            + Sigma1 * cls.M1
            + Sigma2 * cls.M2
            + Sigma3 * cls.M3
        )

        b = Q1 * cls.f1 + Q2 * cls.f2 + Q3 * cls.f3

        return A, b

    def fom_solver(self, cls, param):
        A, b = self.fom_operators(cls, param)

        u = self.basis.zeros()
        u[self.dirichlet_boundary_dofs] = self.dirichlet_boundary_value

        return solve(*condense(A, b, x=u, I=self.free_dofs))

    def reduced_operators(self, cls):
        """
        Precompute reduced affine pieces and mean-shift terms.
        """
        self.U = cls.V_sel.copy()
        self.u_mean = cls.train_ref.copy()

        K1 = _ensure_csr(cls.K1)
        K2 = _ensure_csr(cls.K2)
        K3 = _ensure_csr(cls.K3)
        M1 = _ensure_csr(cls.M1)
        M2 = _ensure_csr(cls.M2)
        M3 = _ensure_csr(cls.M3)

        f1 = np.asarray(cls.f1, float).ravel()
        f2 = np.asarray(cls.f2, float).ravel()
        f3 = np.asarray(cls.f3, float).ravel()

        self.K1_r = self.U.T @ (K1 @ self.U)
        self.K2_r = self.U.T @ (K2 @ self.U)
        self.K3_r = self.U.T @ (K3 @ self.U)
        self.M1_r = self.U.T @ (M1 @ self.U)
        self.M2_r = self.U.T @ (M2 @ self.U)
        self.M3_r = self.U.T @ (M3 @ self.U)

        self.f1_r = self.U.T @ f1
        self.f2_r = self.U.T @ f2
        self.f3_r = self.U.T @ f3

        self.gK1 = self.U.T @ (K1 @ self.u_mean)
        self.gK2 = self.U.T @ (K2 @ self.u_mean)
        self.gK3 = self.U.T @ (K3 @ self.u_mean)
        self.gM1 = self.U.T @ (M1 @ self.u_mean)
        self.gM2 = self.U.T @ (M2 @ self.u_mean)
        self.gM3 = self.U.T @ (M3 @ self.u_mean)

    def rom_solver(self, cls, param):
        """
        Solve the reduced-order model.

        Parameter vector:
            [D1, D2, D3, Sigma1, Sigma2, Sigma3, Q1, Q2, Q3]
        """
        D1, D2, D3, Sigma1, Sigma2, Sigma3, Q1, Q2, Q3 = param

        if cls.cur_itr == 0 or not hasattr(self, "K1_r"):
            self.fom_operators(cls, param)
            self.reduced_operators(cls)

        A_r = (
            D1 * self.K1_r
            + D2 * self.K2_r
            + D3 * self.K3_r
            + Sigma1 * self.M1_r
            + Sigma2 * self.M2_r
            + Sigma3 * self.M3_r
        )

        b_r = (
            Q1 * self.f1_r
            + Q2 * self.f2_r
            + Q3 * self.f3_r
            - D1 * self.gK1
            - D2 * self.gK2
            - D3 * self.gK3
            - Sigma1 * self.gM1
            - Sigma2 * self.gM2
            - Sigma3 * self.gM3
        )

        u_r = np.linalg.solve(A_r, b_r)
        return u_r

    def hyper_rom_solver_ecsw(self):
        pass

    def hyper_rom_solver_ecm(self):
        pass

    def hyper_rom_solver_deim(self):
        pass