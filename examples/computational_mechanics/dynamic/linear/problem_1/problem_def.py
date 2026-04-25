"""
Problem_1: Linear Elasticity in a static setting (Affine)
"""
import os
from skrom.problem_classes.masterclass import register_problem, Problem
import numpy as np
from skrom.rom.rom_utils import _ensure_csr
from skrom.fom.fem_utils import load_domain
from skrom.utils.dynamics.integrators import newmark_with_damping
from skfem import asm
from scipy.sparse import csc_matrix

PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))


@register_problem(PROBLEM_NAME)

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
        from domain import domain_
        return domain_()


    def bilinear_forms(self):
        """
        Collect affine bilinear forms for λ and μ.

        Returns
        -------
        [stiffness_lam, stiffness_mu]
        """
        from bilinear_forms import stiffness_lam, stiffness_mu, mass
        return [stiffness_lam, stiffness_mu, mass]


    def linear_forms(self):
        """
        Collect affine linear form for traction.

        Returns
        -------
        [traction]
        """
        from linear_forms import traction
        return [traction]


    def properties(self):
        """
        Return function mapping (E, ν, region) → (λ, μ).

        Returns
        -------
        callable
            lame_params(E, ν, region)
        """
        from properties import lame_params,damping_params

        return lame_params, damping_params


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
        from params import parameters
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
        return np.linspace(0,40*3,2000)


    # ─────────────────────────────────────────────────────────────
    # Small utilities (kept local to this problem definition)
    # ─────────────────────────────────────────────────────────────
    def _unwrap_item(self, obj):
        """Unwrap objects loaded from np.savez (often 0-d object arrays)."""
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray) and obj.dtype == object and obj.shape == ():
                return obj.item()
        except Exception:
            pass
        return obj

    def _dof_indices(self, dofs):
        """Return DOF indices as a 1D int array for skfem DofsView or array-like."""
        import numpy as _np
        if hasattr(dofs, "all"):
            return _np.asarray(dofs.all(), dtype=int).ravel()
        return _np.asarray(dofs, dtype=int).ravel()


    # ─────────────────────────────────────────────────────────────
    # Small utilities (kept local to this problem definition)
    # ─────────────────────────────────────────────────────────────
    def _unwrap_item(self, obj):
        """Unwrap objects loaded from np.savez (often 0-d object arrays)."""
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray) and obj.dtype == object and obj.shape == ():
                return obj.item()
        except Exception:
            pass
        return obj

    def _dof_indices(self, dofs):
        """Return DOF indices as a 1D int array for skfem DofsView or array-like."""
        import numpy as _np
        if hasattr(dofs, "all"):
            return _np.asarray(dofs.all(), dtype=int).ravel()
        return _np.asarray(dofs, dtype=int).ravel()



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
        Full-order transient 3D linear elasticity (two-material beam).

        Notes
        -----
        - Semi-discrete system on free DOFs:
              M_ff ü + C_ff u̇ + K_ff u = f_ff(t),
          with Rayleigh damping C_ff = c_v M_ff + c_m K_ff.
        - Time integration uses Newmark-β (average acceleration).
        - This solver returns the **final displacement snapshot** u(T) as a vector so
          the standard snapshot/POD workflow operates on a 2D snapshot matrix
          (n_snapshots × n_dofs).
        """

        # Time grid
        times = self.time_vector()
        self.time = times[0]
        n_steps = len(times)

        # Operators (cached in cls on first call)
        M0, K_blocks = self.fom_operators(cls)
        E, nu = param

        # Assemble full stiffness from affine blocks
        K0 = sum(
            block * coef
            for region, mats in K_blocks.items()
            for block, coef in zip(
                mats.values(),
                self.properties()[0](E, nu, region)
            )
        ).tocsr()
        M0 = M0.tocsr()

        # Load full domain info (Dirichlet/Neumann DOFs, bases, etc.)
        load_domain(self)

        # Dirichlet BC vector (constant in time)
        u_bc_full = self.basis.zeros()
        u_bc_full[self.dirichlet_dofs] = self.dirichlet_boundary_value

        # Free vs Dirichlet DOFs
        remove = self._dof_indices(self.dirichlet_dofs)
        all_dofs = np.arange(K0.shape[0])
        free = np.setdiff1d(all_dofs, remove, assume_unique=False)

        # Extract constrained blocks
        M_ff = M0[free][:, free].tocsr()
        K_ff = K0[free][:, free].tocsr()

        # Rayleigh damping
        c_v, c_m = self.properties()[1]()
        C_ff = c_v * M_ff + c_m * K_ff

        # Assemble/load time-dependent force on free DOFs (cached for the first snapshot)
        # Assumption: Neumann load is parameter-independent (may depend on time).
        if cls.cur_itr == 0:
            self.F_free_mat = []

            def _force_free(i, times_):
                self.time = times_[i]
                F_full = self.fom_rhs(cls)  # assembled at current time
                f_free = np.asarray(F_full).ravel()[free]
                self.F_free_mat.append(f_free)
                return f_free
        else:
            def _force_free(i, times_):
                return self.F_free_mat[i]

        # Initial conditions (free DOFs)
        U0 = u_bc_full.copy()
        V0 = np.zeros_like(U0)

        # Time integration
        U_c, V_c, A_c = newmark_with_damping(
            M_ff, C_ff, K_ff,
            _force_free,
            times,
            U0=U0[free], V0=V0[free],
        )

        # Reconstruct full displacement history (optional debugging)
        n_dof = K0.shape[0]
        U_full = np.zeros((n_dof, n_steps))
        U_full[free, :] = U_c
        U_full[remove, :] = self.dirichlet_boundary_value

        # Keep histories on the problem instance (not persisted by ROM_data_gen)
        self._fom_times = times
        self._fom_U_full = U_full

        # Return final displacement snapshot (vector)
        return U_full.T



    def reduced_operators(self, cls):
        """
        Reduced-order model solver for transient 3D linear elasticity.

        Notes
        -----
        - The ROM evolves the mean-subtracted displacement:
              d̃ = U - U_mean,
          where U is the full displacement and U_mean is the mean displacement.
        - The ROM system is:
              M_rom d̃̈ + C_rom d̃̇ + K_rom d̃ = f_rom(t),
          where M_rom, C_rom, K_rom are projected operators.
        """
        load_domain(self)

        # Unwrap cached operators loaded from ROM_data (.npz may store 0-d object arrays)
        K_list = self._unwrap_item(getattr(cls, "K_list", None))
        M0     = self._unwrap_item(getattr(cls, "M", None))
        rhs    = self._unwrap_item(getattr(cls, "rhs_linear", None))

        # If offline data did not persist operators, assemble them now
        if K_list is None or M0 is None:
            M0, K_list = self.fom_operators(cls)
        if rhs is None:
            # Assemble Neumann load at current time (do not depend on cls.cur_itr)
            b_lin = self.linear_forms()[0]
            rhs = asm(b_lin, self.fbasis_neumann, **self.assemble_kwargs())

        self.K_list = K_list
        self.M0 = M0
        self.rhs_linear = rhs

        # DOF partitions
        remove = self._dof_indices(self.dirichlet_dofs)
        all_dofs = np.arange(_ensure_csr(M0).shape[0])
        free = np.setdiff1d(all_dofs, remove, assume_unique=False)
        self._remove = remove
        self._free   = free

        # Mean used by master_class reconstruction
        mean = getattr(cls, "test_ref", None)
        if mean is None:
            mean = getattr(cls, "train_ref", None)
        mean = self._unwrap_item(mean)

        if mean is None:
            raise ValueError("ROM mean (train_ref/test_ref) is missing. Run offline stage first.")

        # # Enforce Dirichlet values on the mean
        # mean = mean.copy()
        # if mean.ndim == 1:
        #     mean[remove] = self.dirichlet_boundary_value
        # else:
        #     mean[remove, ...] = self.dirichlet_boundary_value

        self.mean_full = mean

        # # Basis matrix (enforce Dirichlet constraints)
        V = np.asarray(cls.V_sel, dtype=float).copy()
        # # V[remove, :] = 0.0
        # cls.V_sel = V  # ensure master_class reconstruction uses the constrained basis

        # self.V_full = V
        V_free = V#[free, :]
        self.V_free = V

        # Free-free mass block and reduced mass
        M_ff = _ensure_csr(M0)#[free][:, free].tocsr()
        self.M_ff = M_ff
        self.M_r = V_free.T @ (M_ff @ V_free)

        # Affine reduced stiffness blocks and mean-shift blocks
        mean_free = mean# if mean.ndim == 1 else mean[free, ...]
        self.K_r_a = {}
        self.K_a_Tk = {}

        for region, mats in K_list.items():
            self.K_r_a[region] = {}
            self.K_a_Tk[region] = {}
            for mat, B in mats.items():
                B_ff = _ensure_csr(B)#[free][:, free].tocsr()
                self.K_r_a[region][mat] = V_free.T @ (B_ff @ V_free)
                # Mean-shift term for reduced RHS: V^T K ū
                if mean.ndim == 1:
                    self.K_a_Tk[region][mat] = V_free.T @ (B_ff @ mean_free)
                else:
                    # Snapshot workflow: take last slice if mean is time-dependent
                    self.K_a_Tk[region][mat] = V_free.T @ (B_ff @ mean_free[cls.cur_itr].ravel())

        # Reduced load history (time-dependent traction) is built on first ROM call
        self._f_r_list = None
        self._rom_ready = True
    
    
    
    def rom_solver(self, cls, param):
        """
        Standard Galerkin ROM for transient linear elastodynamics (Newmark-β).

        Returns
        -------
        q_T : ndarray, shape (r,)
            Reduced coordinates at final time T for the mean-subtracted displacement.
            master_class.reconstruct_solution maps this back to full coordinates.
        """
        if not getattr(self, "_rom_ready", False):
            self.reduced_operators(cls)

        # Ensure unwrapped operator dict
        self.K_list = self._unwrap_item(self.K_list)
        K_list = self.K_list

        E, nu = param

        # Parameter coefficients for each region/material block
        props = {
            region: {
                mat: val
                for mat, val in zip(
                    K_list[region].keys(),
                    self.properties()[0](E, nu, region)
                )
            }
            for region in K_list
        }

        # Assemble reduced stiffness and mean-shift term
        K_r = np.zeros_like(next(iter(next(iter(self.K_r_a.values())).values())))
        Tk  = np.zeros(self.V_free.shape[1])

        for rgn in props:
            for mat in props[rgn]:
                coef = props[rgn][mat]
                K_r += coef * self.K_r_a[rgn][mat]
                Tk  += coef * self.K_a_Tk[rgn][mat]

        # Rayleigh damping in reduced space
        c_v, c_m = self.properties()[1]()

        C_r = c_v * self.M_r + c_m * K_r

        # Time grid
        times = self.time_vector()

        # Build reduced force history once (assumes traction is parameter-independent)
        if self._f_r_list is None:
            f_free_hist = None
            if hasattr(cls, "F_free_mat"):
                try:
                    f_free_hist = self._unwrap_item(cls.F_free_mat)
                except Exception:
                    f_free_hist = cls.F_free_mat

            if f_free_hist is None:
                load_domain(self)
                free = self._free
                f_free_hist = []
                for i, t in enumerate(times):
                    self.time = t
                    # Assemble Neumann load directly (do not depend on cls.cur_itr)
                    b_lin = self.linear_forms()[0]
                    F_full = asm(b_lin, self.fbasis_neumann, **self.assemble_kwargs())
                    f_free_hist.append(np.asarray(F_full).ravel())#[free])

            self._f_r_list = [self.V_free.T @ f for f in f_free_hist]

        # Reduced forcing callback (mean-shift appears as constant subtraction)
        def force_red(i, times_):
            return self._f_r_list[i] - Tk

        # Initial conditions for mean-subtracted field: u_ms(0) = u(0) - ū
        free = self._free
        remove = self._remove

        u0_full = np.zeros(_ensure_csr(self.M0).shape[0])
        u0_full[remove] = self.dirichlet_boundary_value
        u0_free = u0_full#[free]

        mean = self.mean_full
        mean_free = mean# if mean.ndim == 1 else mean[free, ...]
        if mean.ndim != 1:
            mean_free = mean_free.ravel()

        u_ms0 = u0_free - mean_free
        q0 = self.V_free.T @ u_ms0
        qd0 = np.zeros_like(q0)

        # Time integration in reduced space
        Q, Qd, Qdd = newmark_with_damping(
            self.M_r, C_r, K_r,
            force_red,
            times,
            U0=q0, V0=qd0,
        )

        # Store history for optional post-processing
        self._rom_times = times
        self._rom_Q = Q

        # Return final reduced coordinates (snapshot ROM)
        return Q



    def hyper_rom_solver_ecsw(self):
        """Solve hyper-reduced-order model for given parameters."""
        pass
    
    def hyper_rom_solver_deim(self):
        """Solve hyper-reduced-order model for given parameters."""
        pass

    def hyper_rom_solver_ecm(self):
        """Solve hyper-reduced-order model for given parameters."""
        pass