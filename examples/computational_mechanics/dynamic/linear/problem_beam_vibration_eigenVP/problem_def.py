"""
Problem: Two-segment cantilever beam vibration (Timoshenko model).

Full-order model (FOM):
- Assemble parameter-independent blocks per region
- Build K(h1,h2) and M(h1,h2)
- Solve generalized eigenproblem:  K u = lambda M u
- Return first mode shape u (stacked as [w; theta])

Reduced-order model (ROM):
- Project blocks onto POD basis U
- Solve reduced eigenproblem:  K_r a = lambda M_r a
- Return reduced coordinates a (mode coefficients)
"""

import os
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.linalg import eigh

from skfem import asm
from skrom.fom.fem_utils import load_domain
from skrom.problem_classes.static.master_class import register_problem, Problem

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))


@register_problem(PROBLEM_NAME)
class ProblemBeamVibration(Problem):

    def domain(self):
        from domain import domain_
        return domain_()

    def bilinear_forms(self):
        from bilinear_forms import a_grad, a_mass, a_shear_w_theta, a_shear_theta_w
        return [a_grad, a_mass, a_shear_w_theta, a_shear_theta_w]

    def linear_forms(self):
        from linear_forms import l_zero
        return [l_zero]

    def properties(self):
        from properties import beam_coeffs
        return [beam_coeffs]

    def parameters(self, n_samples):
        from params import parameters
        return parameters(n_samples)

    # ---------- Assembly caches ----------

    def fom_operators(self, cls):
        """
        Cache parameter-independent blocks (per region).
        """
        if cls.cur_itr == 0:
            load_domain(self)

            a_grad, a_mass, a_wt, a_tw = self.bilinear_forms()

            # Region-wise scalar blocks
            cls.G_list = {r: asm(a_grad, b) for r, b in self.basis_regions.items()}
            cls.M_list = {r: asm(a_mass, b) for r, b in self.basis_regions.items()}

            cls.C_wt_list = {r: asm(a_wt, b) for r, b in self.basis_regions.items()}
            cls.C_tw_list = {r: asm(a_tw, b) for r, b in self.basis_regions.items()}

            # Bookkeeping
            cls.n_scalar = int(self.n_scalar_dofs)

            # Root BC indices for scalar field; full system uses [w; theta] stacking
            cls.dir0 = np.asarray(self.dirichlet_boundary_dofs_scalar, dtype=int)

            # Tip DOF for sign convention
            cls.tip0 = int(np.asarray(self.right_boundary_dofs_scalar, dtype=int)[0])

            # Store frequencies
            cls.omega_fos = []

        return cls.G_list, cls.M_list, cls.C_wt_list, cls.C_tw_list

    # ---------- Full-order solver ----------

    def fom_solver(self, cls, param):
        """
        Return the first mode shape (stacked [w; theta]) for the current parameters.
        """
        (G_list, M_list, C_wt_list, C_tw_list) = self.fom_operators(cls)

        h1, h2 = float(param[0]), float(param[1])
        beam_coeffs = self.properties()[0]

        n = cls.n_scalar

        # Accumulate scalar blocks for each sub-block in K and M
        Kww = sparse.csr_matrix((n, n))
        Kwth = sparse.csr_matrix((n, n))
        Kthw = sparse.csr_matrix((n, n))
        Kthth = sparse.csr_matrix((n, n))

        Mww = sparse.csr_matrix((n, n))
        Mthth = sparse.csr_matrix((n, n))

        for region in G_list:
            EI, kGA, rhoA, rhoI = beam_coeffs(h1, h2, region)

            G = G_list[region]
            Ms = M_list[region]
            C_wt = C_wt_list[region]
            C_tw = C_tw_list[region]

            # Stiffness
            Kww += kGA * G
            Kwth += kGA * C_wt
            Kthw += kGA * C_tw
            Kthth += EI * G + kGA * Ms

            # Mass
            Mww += rhoA * Ms
            Mthth += rhoI * Ms

        K = sparse.bmat([[Kww, Kwth],
                         [Kthw, Kthth]], format="csr")
        M = sparse.bmat([[Mww, None],
                         [None, Mthth]], format="csr")

        # Dirichlet BCs at x=0 for both w and theta
        dir_w = cls.dir0
        dir_th = cls.dir0 + n
        dir_full = np.concatenate([dir_w, dir_th])

        all_dofs = np.arange(2 * n, dtype=int)
        free_full = np.setdiff1d(all_dofs, dir_full)

        Kff = K[free_full][:, free_full]
        Mff = M[free_full][:, free_full]

        # Generalized eigenproblem: Kff x = lam Mff x (lowest mode)
        # Use shift-invert near zero for robustness.
        vals, vecs = spla.eigsh(Kff, k=1, M=Mff, sigma=0.0, which="LM")
        lam = float(vals[0])
        x = vecs[:, 0]

        # Reconstruct full vector
        u = np.zeros(2 * n)
        u[free_full] = x
        u[dir_full] = 0.0

        # Normalize using max |w|
        w = u[:n]
        maxw = np.max(np.abs(w))
        if maxw > 0:
            u /= maxw

        # Fix sign so that w(L) >= 0
        if u[cls.tip0] < 0:
            u *= -1.0

        # Store natural frequency (rad/s)
        cls.omega_fos.append(np.sqrt(lam))

        return u

    # ---------- Reduced operators and ROM solver ----------

    def reduced_operators(self, cls):
        """
        Project region-wise blocks onto reduced basis U.
        """
        self.U = cls.V_sel

        # unwrap cached lists from disk if needed
        G_list   = cls.G_list.item()   if hasattr(cls.G_list, "item") else cls.G_list
        M_list   = cls.M_list.item()   if hasattr(cls.M_list, "item") else cls.M_list
        C_wt_list = cls.C_wt_list.item() if hasattr(cls.C_wt_list, "item") else cls.C_wt_list
        C_tw_list = cls.C_tw_list.item() if hasattr(cls.C_tw_list, "item") else cls.C_tw_list

        n = int(cls.n_scalar)

        # Build region-wise full block matrices once, then project
        self.Kbend_r = {}
        self.Kshear_r = {}
        self.Mtrans_r = {}
        self.Mrot_r = {}

        for region in G_list:
            G = G_list[region]
            Ms = M_list[region]
            C_wt = C_wt_list[region]
            C_tw = C_tw_list[region]

            # Full matrices for unit coefficients
            Z = sparse.csr_matrix((n, n))

            Kshear = sparse.bmat([[G, C_wt],
                                  [C_tw, Ms]], format="csr")
            Kbend  = sparse.bmat([[Z, Z],
                                [Z, G]], format="csr")
            Mtrans = sparse.bmat([[Ms, Z],
                                [Z,  Z]], format="csr")
            Mrot   = sparse.bmat([[Z,  Z],
                                [Z,  Ms]], format="csr")

            # Project
            self.Kbend_r[region]  = self.U.T @ (Kbend @ self.U)
            self.Kshear_r[region] = self.U.T @ (Kshear @ self.U)
            self.Mtrans_r[region] = self.U.T @ (Mtrans @ self.U)
            self.Mrot_r[region]   = self.U.T @ (Mrot @ self.U)

        # For sign normalization in ROM
        self.tip_full = int(cls.tip0)           # w-tip index in full vector
        self.n_scalar = int(n)

    def rom_solver(self, cls, param):
        """
        Solve reduced generalized eigenproblem and return reduced mode coefficients.
        """
        h1, h2 = float(param[0]), float(param[1])
        beam_coeffs = self.properties()[0]

        if cls.cur_itr == 0:
            self.reduced_operators(cls)

        # Assemble reduced K and M
        K_r = np.zeros((self.U.shape[1], self.U.shape[1]))
        M_r = np.zeros((self.U.shape[1], self.U.shape[1]))

        for region in self.Kbend_r:
            EI, kGA, rhoA, rhoI = beam_coeffs(h1, h2, region)

            K_r += EI * self.Kbend_r[region] + kGA * self.Kshear_r[region]
            M_r += rhoA * self.Mtrans_r[region] + rhoI * self.Mrot_r[region]

        # Solve reduced eigenproblem (dense)
        vals, vecs = eigh(K_r, M_r)
        lam = float(vals[0])
        a = vecs[:, 0]

        # Normalize and fix sign based on reconstructed w(L)
        u_full = self.U @ a
        w = u_full[:self.n_scalar]
        maxw = np.max(np.abs(w))
        if maxw > 0:
            a = a / maxw
            u_full = u_full / maxw

        if u_full[self.tip_full] < 0:
            a = -a

        # Store omega if desired
        if not hasattr(cls, "omega_rom"):
            cls.omega_rom = []
        cls.omega_rom.append(np.sqrt(lam))

        return a

    def hyper_rom_solver_ecsw(self):
        pass

    def hyper_rom_solver_deim(self):
        pass
