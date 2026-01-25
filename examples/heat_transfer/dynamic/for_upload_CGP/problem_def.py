from skrom.problem_classes.static.master_class import register_problem, Problem
import numpy as np
from skrom.rom.rom_utils import _ensure_csr
from skrom.fom.fem_utils import load_domain, newton_solver
from skfem import *
from scipy.sparse import csc_matrix
import os
from pathlib import Path
from skrom.rom.bilinear_form_rom import BilinearFormROM
from skrom.rom.linear_form_rom  import LinearFormROM

from skrom.rom.ecsw.bilinear_form_hyperrom_ecsw import BilinearFormHYPERROM_ecsw
from skrom.rom.ecsw.linear_form_hyperrom_ecsw  import LinearFormHYPERROM_ecsw
from skrom.rom.rom_utils import newton_hyper_rom_solver2, reconstruct_solution
import threading
# _mutex = threading.Lock()


# Derive the problem name from this file’s parent folder
PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))


@register_problem(PROBLEM_NAME)
class Problem3Dprinting(Problem):
    """
    Template for an affine or non-linear reduced-order-model (ROM) problem.
    """

    # -------------------------------
    # Thread-safe state helpers
    # -------------------------------
    _domain_lock = threading.Lock()
    _domain_loaded = False
    _thread_local = threading.local()

    def _tls(self):
        """Per-thread storage for per-snapshot/per-param objects."""
        return self._thread_local

    def _ensure_domain_loaded(self):
        """Load mesh/basis/DOFs once per Problem instance (thread-safe)."""
        if getattr(self, "_domain_loaded", False):
            return
        with self._domain_lock:
            if getattr(self, "_domain_loaded", False):
                return
            load_domain(self)
            self._domain_loaded = True

    # ------------------------------------------------------------------
    # Geometry & discretisation
    # ------------------------------------------------------------------

    def domain(self):
        """
        Import domain information from **domain.py** in the local directory.
        No geometry is built here – we simply delegate to *domain_*.

        Example
        -------
        from domain import domain_
        return domain_()

        Required keys (but not limited to) in the returned dict:
        * 'mesh', 'basis'
        * 'free_dofs', 'dirichlet_dofs', 'dirichlet_value'
        """

        from domain import domain_3d
        return domain_3d()

    # ------------------------------------------------------------------
    # Weak forms
    # ------------------------------------------------------------------

    def bilinear_forms(self):
        """
        Import element-level bilinear (or Jacobian) forms from
        **bilinear_forms.py**.  Nothing is assembled here – we merely hand back
        the callables.

        Example
        -------
        >>> from bilinear_forms import a1, a2
        >>> return [a1, a2]
        """
        
        from bilinear_forms import mass_form_bil, laplace_form_bil, jacobian, jac_bnd, jac_bnd_only_rad

        return [mass_form_bil, laplace_form_bil, jacobian, jac_bnd, jac_bnd_only_rad]

    def linear_forms(self):
        """
        Import element-level linear / residual forms from **linear_forms.py**.
        No assembly happens here – we just return the callables.

        Example
        -------
        >>> from linear_forms import f1, f2
        >>> return [f1, f2]
        """

        from linear_forms import mass_form, laplace_form, convection_radiation_bdd, convection_bdd, radiation_bdd, make_laser_flux, radiation_bdd_hyp
        return [mass_form, laplace_form, convection_radiation_bdd, convection_bdd, radiation_bdd, make_laser_flux, radiation_bdd_hyp]

    # ------------------------------------------------------------------
    # Material / source coefficients
    # ------------------------------------------------------------------

    def properties(self):
        """
        Import coefficient-generating functions (e.g. *k(μ)*, *q(β)*, …) from
        **properties.py** located in the same folder.

        Example
        -------
        >>> from properties import k_func, q_func
        >>> return [k_func, q_func]
        """
        from properties import material_properties
        return material_properties

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------

    def parameters(self, n_samples):
        """
        Import a sampling-design generator from **params.py**.  The helper
        function constructs training / test parameter sets.

        Example
        -------
        >>> from params import parameters
        >>> return parameters(n_samples)
        """
        from params import parameters
        return parameters(n_samples)

    def assemble_kwargs_laser(self, cls, time=None):
        """
        Pack E, ν into kwargs for assembly routines.

        Parameters
        ----------
        param : tuple (E, ν)

        Returns
        -------
        dict
            {'E': E, 'nu': ν}
        """
        laser_dict = cls.cases[0]
        traj =laser_dict['traj_class'][0](cls.dt, laser_dict['feed_rate'][0], **laser_dict['traj_kwargs'][0])
        

        return dict(time=time,
                    traj=traj,
                    # Qp=laser_dict["Qp"][0],
                    eta=laser_dict["eta"][0],
                    r=laser_dict["r"][0])

    def assemble_kwargs_hyperreduction(self, u):
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
            {'sol'}.
        """
        return dict(sol=u)


    # ------------------------------------------------------------------
    # Full-order model (FOM)
    # ------------------------------------------------------------------

    def fom_operators(self, cls):

        ## Load everything related to the problem ONCE IN FOR ALL

        load_domain(self)  # init mesh, basis, DOFs
        from properties import material_properties
        from params import simulation_params

        cls.dt, cls.nsteps, cls.t_end, cls.ts, cls.theta, cls.cases = simulation_params()
     
        self.ts = np.linspace(0, cls.t_end, cls.nsteps + 1)

        cls.rho, cls.cp, cls.k, cls.T_infty,  cls.h,  cls.Rboltz,  cls.emiss = material_properties()


        non_dirichlet = {'left', 'right', 'front', 'back', 'top'}
        cls.fbasis_non_dir = FacetBasis(self.mesh, self.element, facets=non_dirichlet)
        cls.fbasis_top     = FacetBasis(self.mesh, self.element, facets='top')
        cls.bottom_dofs    = self.basis.get_dofs('bottom')

        # cls.traj = cls.cases[0]['traj_class'][0](cls.dt, cls.cases[0]['feed_rate'][0], **cls.cases[0]['traj_kwargs'][0])


        mass_form_bil, laplace_form_bil, jacobian = self.bilinear_forms()[0], self.bilinear_forms()[1], self.bilinear_forms()[2]

        # Precompute bilinear and Jacobian forms

        cls.M_bilin = asm(mass_form_bil, self.basis)
        cls.K_bilin = asm(laplace_form_bil, self.basis)
        cls.Jac     = asm(jacobian, self.basis)

        return cls.M_bilin, cls.K_bilin, cls.Jac


    def fom_operators_t(self, cls, u_prev_bdd):

        self.jac_bnd = self.bilinear_forms()[3]
        J_bnd = asm(self.jac_bnd, cls.fbasis_non_dir, prev=u_prev_bdd)

        return J_bnd


    def fom_rhs_t(self, cls, t,  u_cur, u_old, convection_radiation_bdd, laser_flux_form):
        """
        Assemble (and cache) the full-order RHS vector consumed by
        **fom_solver** and hyper-reduction routines.

        Parameters
        ----------
        cls : master_class object
            Simulation context (see *fom_operators* docstring).
        """

        # kw.update({"time":t})
        # bc_laser = asm(laser_flux, cls.fbasis_top, **kw)

        bc_laser = asm(laser_flux_form, cls.fbasis_top, time=t)


        M_lin = cls.M_bilin.dot((u_cur - u_old) / cls.dt)
        K_lin = cls.K_bilin.dot(cls.theta * u_old + (1 - cls.theta) * u_cur)
        u_prev_bdd  = cls.fbasis_non_dir.interpolate(u_old)
        u_trial_bdd = cls.fbasis_non_dir.interpolate(u_cur)
        bc_neum = asm(convection_radiation_bdd,
                    cls.fbasis_non_dir,
                    trial=u_trial_bdd,
                    prev=u_prev_bdd)
        
        RHS = M_lin + K_lin - bc_neum - bc_laser

        return RHS


    def fom_solver(self, cls, param):
        """
        Solve the high-fidelity model for one parameter point.

        Called automatically by the master class when a simulation is run.

        Parameters
        ----------
        cls : master_class object
            Contains run-time info such as **cls.cur_itr**.
        param : ndarray or scalar
            Parameter vector/value μ at which to solve.

        Returns
        -------
        full_solution : ndarray
        """

        # Initial condition

        self.fom_operators(cls)

        make_laser_flux = self.linear_forms()[-2]
        convection_radiation_bdd = self.linear_forms()[2]

        kw  = self.assemble_kwargs_laser(cls)

        laser_flux_form = make_laser_flux(
            traj=kw["traj"],   # your trajectory object
            Qp=param,
            eta=kw["eta"],
            r=kw["r"],
        )

        u_old = np.full(cls.basis.N, cls.T_infty)
        u_sol = []
        u_sol.append(u_old.copy())

        # Time‐marching loop
        time_steps = self.ts
        assemble_args ={}


        def assemble_fn(u_old):

            u_prev_bdd  = cls.fbasis_non_dir.interpolate(u_old)
            J_bnd = self.fom_operators_t(cls, u_prev_bdd)

            return cls.Jac - J_bnd
        
        
        def rhs_fn(u_cur, u_old, t):

            # kw = self.assemble_kwargs_laser(cls)
            return self.fom_rhs_t(cls, t, u_cur, u_old, convection_radiation_bdd, laser_flux_form)



        for i in range(len(cls.ts) - 1):

            t_theta = cls.theta * time_steps[i] + (1 - cls.theta) * time_steps[i + 1]


            u_new = newton_solver(
                assemble_fn,
                rhs_fn,
                u_old,
                cls.bottom_dofs,
                cls.T_infty,
                *assemble_args,
                tol = 1e-3,
                maxit = 50,
                alpha = 1.0,
                # NEW: extra RHS args come after assemble args via explicit keyword
                rhs_args = (u_old, t_theta),
                jac_conditioner=True,
                # PETSc KSP options
                ksp_type="gmres",
                pc_type="ilu",
                ksp_rtol=1e-8,
                ksp_max_it=2000,
                reuse_ksp=True,
            )

            u_old = u_new.copy()
            u_sol.append(u_new.copy())

            if i % 10 == 0:
                print(f"{i=}")

        return np.array(u_sol)


    # ------------------------------------------------------------------
    # Hyper-reduction (ECSW / DEIM)
    # ------------------------------------------------------------------


    def hyper_rom_operators_ecsw(self, cls):
        tls = self._tls()

        mass_form_bil        = self.bilinear_forms()[0]
        laplace_form_bil     = self.bilinear_forms()[1]
        jacobian             = self.bilinear_forms()[2]
        jac_bnd_only_rad     = self.bilinear_forms()[4]

        # Project constant operators (use .item() because your stored matrices may be 0-d object arrays)
        M_full   = cls.M_bilin.item() if hasattr(cls.M_bilin, "item") else cls.M_bilin
        K_full   = cls.K_bilin.item() if hasattr(cls.K_bilin, "item") else cls.K_bilin
        Jac_full = cls.Jac.item()     if hasattr(cls.Jac, "item")     else cls.Jac

        tls.M_bilin_rom = tls.U.T @ M_full   @ tls.U
        tls.K_bilin_rom = tls.U.T @ K_full   @ tls.U
        tls.Jac_rom     = tls.U.T @ Jac_full @ tls.U

        # Convection (ROM-assembled)
        mass_form_bil_rom_conv = BilinearFormROM(
            mass_form_bil, self.fbasis_non_dirichlet, tls.U, tls.U,
            free_dofs=self.free_dofs
        )
        tls.M_bilin_rom_conv = mass_form_bil_rom_conv.assemble()

        # Radiation Jacobian on boundary (ECSW)
        tls.jac_rad_bnd_bil_rom = BilinearFormHYPERROM_ecsw(
            jac_bnd_only_rad, tls.weights, self.fbasis_non_dirichlet,
            tls.U, tls.U, free_dofs=self.free_dofs
        )

        # Linear forms
        convection_bdd, radiation_bdd = self.linear_forms()[3], self.linear_forms()[4]

        if cls.cur_itr==0 and not Path("bc_laser_ecsw_list.npy").exists():

            tls.laser_flux_lin_rom = LinearFormROM(
                tls.laser_flux_form, self.fbasis_top, tls.U, free_dofs=self.free_dofs
            )

        convection_bdd_lin_rom_form = LinearFormROM(
            convection_bdd, self.fbasis_non_dirichlet, tls.U, free_dofs=self.free_dofs
        )
        tls.bc_neumann_lin_rom_conv = convection_bdd_lin_rom_form.assemble()

        tls.radiation_bdd_lin_ROM = LinearFormHYPERROM_ecsw(
            radiation_bdd, tls.weights, self.fbasis_non_dirichlet,
            tls.U, free_dofs=self.free_dofs
        )



    def hyper_rom_rhs_ecsw_t(self, u_cur_rom, u_old_rom, t, cls, i, param):
        tls = self._tls()
        bc_file = Path("bc_laser_ecsw_list.npy")


        # Reconstruct full fields
        u_reconst_cur = reconstruct_solution(u_cur_rom, tls.U, tls.u_ref)
        u_reconst_old = reconstruct_solution(u_old_rom, tls.U, tls.u_ref)

        M_lin_rom = tls.M_bilin_rom.dot((u_cur_rom - u_old_rom) / cls.dt)

        K_lin_rom = (
            tls.K_bilin_rom.dot(cls.theta * u_old_rom + (1 - cls.theta) * u_cur_rom)
            + tls.u_arr_mean_rom
        )

        M_lin_rom_conv = cls.h * (
            tls.M_bilin_rom_conv.dot(cls.theta * u_old_rom + (1 - cls.theta) * u_cur_rom)
            + tls.u_arr_mean_rom_conv
        )

        u_reconst_prev_bdd  = self.fbasis_non_dirichlet.interpolate(u_reconst_old.flatten())
        u_reconst_trial_bdd = self.fbasis_non_dirichlet.interpolate(u_reconst_cur.flatten())

        bc_neumann_lin_rom_rad = tls.radiation_bdd_lin_ROM.assemble_weighted_ecsw(
            trial=u_reconst_trial_bdd, prev=u_reconst_prev_bdd
        )

        if cls.cur_itr==0 and not bc_file.exists():
            # with _mutex:
            bc_laser_lin_rom = tls.laser_flux_lin_rom.assemble(time=t)
            tls.bc_laser_lin_rom_ = bc_laser_lin_rom
        else:
            bc_laser_lin_rom = tls.bc_laser_lin_rom_list[i]

        RHS_rom = (
            M_lin_rom
            + K_lin_rom
            - bc_laser_lin_rom*param
            - (M_lin_rom_conv + tls.bc_neumann_lin_rom_conv)
            - bc_neumann_lin_rom_rad
        )

        return RHS_rom


    def hyper_rom_operator_ecsw_t(self, u_old_rom, cls):
        tls = self._tls()


        u_reconst_old = reconstruct_solution(u_old_rom, tls.U, tls.u_ref)
        u_reconst_prev_bdd = self.fbasis_non_dirichlet.interpolate(u_reconst_old.flatten())

        Jac_boundary_rom = tls.jac_rad_bnd_bil_rom.assemble_weighted_ecsw(prev=u_reconst_prev_bdd)

        J_rom = (
            tls.Jac_rom
            - Jac_boundary_rom
            - tls.M_bilin_rom_conv * cls.h * (1 - cls.theta)
        )

        return J_rom


    def hyper_rom_solver_ecsw(self, cls, param):
        tls = self._tls()


        # Ensure mesh/basis/DOFs exist for this Problem instance
        self._ensure_domain_loaded()

        # Per-snapshot inputs (thread-local)
        tls.U = cls.V_sel
        tls.weights = cls.z

        # Per-snapshot mean / reference
        tls.u_ref = cls.test_ref[cls.cur_itr]

        # Build per-param laser form (thread-local)
        make_laser_flux = self.linear_forms()[-2]
        kw = self.assemble_kwargs_laser(cls)

        bc_file = Path("bc_laser_ecsw_list.npy")

        if cls.cur_itr == 0 and not bc_file.exists():
            # with _mutex:
            tls.bc_laser_lin_rom_list = []
            tls.laser_flux_form = make_laser_flux(
                traj=kw["traj"],
                Qp=1,
                eta=kw["eta"],
                r=kw["r"]
            )
        else:
            tls.bc_laser_lin_rom_list = np.load(bc_file)  # add allow_pickle=True if needed

        # Assemble ROM/hyper-ROM operators (thread-local storage)
        self.hyper_rom_operators_ecsw(cls)

        # Precompute mean shifts (thread-local)
        K_full = cls.K_bilin.item() if hasattr(cls.K_bilin, "item") else cls.K_bilin
        M_full = cls.M_bilin.item() if hasattr(cls.M_bilin, "item") else cls.M_bilin

        u_arr_mean_vec      = K_full @ tls.u_ref.flatten()
        u_arr_mean_conv_vec = M_full @ tls.u_ref.flatten()

        tls.u_arr_mean_rom      = tls.U.T @ u_arr_mean_vec
        tls.u_arr_mean_rom_conv = tls.U.T @ u_arr_mean_conv_vec

        # Time marching in reduced space
        u_old_rom = np.zeros(tls.U.shape[1])
        u_sol_rom = []
        u_sol_rom.append(u_old_rom.copy())

        time_steps = cls.ts
        assemble_args = (cls,)  # tuple

        for i in range(len(cls.ts) - 1):
            t_theta = cls.theta * time_steps[i] + (1 - cls.theta) * time_steps[i + 1]

            u_new = newton_hyper_rom_solver2(
                self.hyper_rom_operator_ecsw_t,
                self.hyper_rom_rhs_ecsw_t,
                u_old_rom,
                *assemble_args,
                tol=1e-3,
                maxit=50,
                alpha=1.0,
                rhs_args=(u_old_rom, t_theta, cls, i, param),
                linear_backend="petsc",
                verbose=False,
            )

            if cls.cur_itr == 0 and not bc_file.exists():
                tls.bc_laser_lin_rom_list.append(tls.bc_laser_lin_rom_.copy())

            u_old_rom = u_new.copy()
            u_sol_rom.append(u_new.copy())
            if i % 10 == 0:
                print(f"[Hyper-ROM ECSW] Time step {i=}")

        if cls.cur_itr == 0 and not bc_file.exists():
            np.save("bc_laser_ecsw_list.npy", np.array(tls.bc_laser_lin_rom_list))

        return np.array(u_sol_rom)



    def rom_solver(self):
        pass
    
    def reduced_operators(self):
        pass

    def hyper_rom_operators(self):
        pass

    def hyper_rom_solver_deim(self):
        pass
    