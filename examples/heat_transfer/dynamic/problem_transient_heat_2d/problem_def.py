"""\
Problem: 2-D transient heat conduction on a unit square.

FOM (implicit Euler):
    M (T^{n+1} - T^{n})/dt + k K T^{n+1} = q f

ROM (Galerkin in a POD basis, with time-dependent mean subtraction):
    Let T(t) \approx \bar{T}(t) + U a(t)

    M_r (a^{n+1} - a^{n})/dt + k K_r a^{n+1} = q f_r - (M_r \dot{\bar{a}} + k K_r \bar{a})

    where \bar{T}(t) is the time-dependent training mean field (stored as train_ref).

Outputs
-------
Both FOM and ROM return the full temperature history as an array of shape (nt, Nh).
"""

from __future__ import annotations

import os
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from skfem import asm

from skrom.fom.fem_utils import load_domain
from skrom.problem_classes.static.master_class import register_problem, Problem

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))


@register_problem(PROBLEM_NAME)
class ProblemTransientHeat2D(Problem):

    def domain(self):
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        from bilinear_forms import a_mass, a_diffusion
        return [a_mass, a_diffusion]

    def linear_forms(self):
        from linear_forms import l_source
        return [l_source]

    def properties(self):
        from properties import k, q
        return [k, q]

    def parameters(self, n_samples):
        from params import parameters
        return parameters(n_samples)

    # -----------------
    # Assembly caches
    # -----------------
    def fom_operators(self, cls):
        """Assemble parameter-independent operators once."""
        if cls.cur_itr == 0:
            load_domain(self)

            a_mass, a_diff = self.bilinear_forms()
            l_src = self.linear_forms()[0]

            # Assemble per-region blocks (single region by default)
            cls.M_list = {r: asm(a_mass, b) for r, b in self.basis_regions.items()}
            cls.K_list = {r: asm(a_diff, b) for r, b in self.basis_regions.items()}
            cls.f_list = {r: asm(l_src, b) for r, b in self.basis_regions.items()}

            # Convenience
            cls.Nh = int(self.basis.N)

            # Persist time-grid information (needed by the ROM stage)
            cls.dt = float(self.dt)
            cls.nt = int(self.nt)
            cls.t_end = float(self.t_end)
            cls.t = np.asarray(self.t)

        return cls.M_list, cls.K_list, cls.f_list

    # -----------------
    # Full-order solver
    # -----------------
    def fom_solver(self, cls, param):
        """Implicit Euler time integration for the full-order model."""
        M_list, K_list, f_list = self.fom_operators(cls)

        k_param, q_param = float(param[0]), float(param[1])
        k_fun, q_fun = self.properties()

        # Global operators
        M = sparse.csr_matrix((cls.Nh, cls.Nh))
        K = sparse.csr_matrix((cls.Nh, cls.Nh))
        f = np.zeros(cls.Nh)
        for region in M_list:
            M += 1.0 * M_list[region]
            K += k_fun(k_param, region) * K_list[region]
            f += q_fun(q_param, region) * f_list[region]

        dt = float(self.dt)
        nt = int(self.nt)

        # Dirichlet vector
        T_dir = np.zeros(cls.Nh)
        T_dir[self.dirichlet_boundary_dofs] = float(self.dirichlet_boundary_value)

        # Initial condition (consistent with Dirichlet)
        Tn = T_dir.copy()

        # Pre-factorize reduced (free) system
        A = (M / dt) + K
        free = np.asarray(self.free_dofs, dtype=int)

        Aff = A[free][:, free].tocsc()
        solve_Aff = spla.factorized(Aff)

        M_over_dt = (M / dt).tocsr()

        # History
        Thist = np.zeros((nt, cls.Nh))
        Thist[0, :] = Tn

        for n in range(nt - 1):
            rhs = (M_over_dt @ Tn) + f

            # Apply Dirichlet contribution if nonzero
            if np.any(T_dir != 0.0):
                rhs = rhs - A @ T_dir

            # Solve for free DOFs
            Tf = solve_Aff(rhs[free])
            Tnp1 = T_dir.copy()
            Tnp1[free] = Tf

            Thist[n + 1, :] = Tnp1
            Tn = Tnp1

        return Thist

    # -----------------
    # Reduced operators
    # -----------------
    def reduced_operators(self, cls):
        """Project full operators onto the selected POD basis."""
        self.U = cls.V_sel

        # Unwrap cached dicts (stored via np.savez)
        M_list = cls.M_list.item() if hasattr(cls.M_list, "item") else cls.M_list
        K_list = cls.K_list.item() if hasattr(cls.K_list, "item") else cls.K_list
        f_list = cls.f_list.item() if hasattr(cls.f_list, "item") else cls.f_list

        # Global unit operators
        Nh = int(cls.Nh)
        M = sparse.csr_matrix((Nh, Nh))
        K0 = sparse.csr_matrix((Nh, Nh))
        f0 = np.zeros(Nh)
        for region in M_list:
            M += 1.0 * M_list[region]
            K0 += 1.0 * K_list[region]
            f0 += 1.0 * f_list[region]
        self.K0=K0.copy()
        # Reduced matrices
        self.M_r = self.U.T @ (M @ self.U)
        self.K_r0 = self.U.T @ (K0 @ self.U)
        self.f_r0 = self.U.T @ f0

        # # Mean terms (time-dependent)
        # mean = cls.train_ref
        # if mean.ndim != 2:
        #     raise ValueError("Expected train_ref to be a 2D array of shape (nt, Nh) for this transient example.")


    # -----------------
    # ROM solver
    # -----------------
    def rom_solver(self, cls, param):
        """Implicit Euler time integration in the reduced coordinates."""
        k_param, q_param = float(param[0]), float(param[1])
        k_fun, q_fun = self.properties()

        # Unwrap cached dicts
        cls.M_list = cls.M_list.item() if hasattr(cls.M_list, "item") else cls.M_list
        cls.K_list = cls.K_list.item() if hasattr(cls.K_list, "item") else cls.K_list
        cls.f_list = cls.f_list.item() if hasattr(cls.f_list, "item") else cls.f_list

        if cls.cur_itr == 0:
            self.reduced_operators(cls)
        
        test_ref = cls.test_ref[cls.cur_itr]
        self.K_mean_r0 = (self.U.T @ (self.K0 @ test_ref.T)).T


        dt = float(cls.dt)
        nt = int(cls.nt)

        # Coefficients (single region, but keep signature consistent)
        k = k_fun(k_param, "region_1")
        q = q_fun(q_param, "region_1")

        Mr = np.asarray(self.M_r)
        Kr = np.asarray(self.K_r0) * k

        A = (Mr / dt) + Kr
        Ainv = np.linalg.inv(A)

        # Precompute time-dependent forcing in reduced coordinates
        # g(t) = q f_r0 - (M_mean_dot_r + k K_mean_r0)
        g = (q * self.f_r0)[None, :] - (k * self.K_mean_r0)

        a_hist = np.zeros((nt, Mr.shape[0]))

        for n in range(nt - 1):
            rhs = (Mr / dt) @ a_hist[n, :] + g#[n + 1, :]
            a_hist[n + 1, :] = Ainv @ rhs.flatten()

        return a_hist

    def hyper_rom_solver_ecsw(self, cls, param):
        raise NotImplementedError("Hyper-reduction is intentionally omitted in this example.")

    def hyper_rom_solver_deim(self, cls, param):
        raise NotImplementedError("Hyper-reduction is intentionally omitted in this example.")
