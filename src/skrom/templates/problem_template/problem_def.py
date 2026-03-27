import numpy as np
import os
from skrom.problem_classes.static.master_class import register_problem, Problem

# Derive the problem name from this file’s parent folder
PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))


@register_problem(PROBLEM_NAME)
class ProblemTemplate(Problem):
    """
    Template for an affine or non-linear reduced-order-model (ROM) problem.
    """

    # ------------------------------------------------------------------
    # Geometry & discretisation
    # ------------------------------------------------------------------
    def domain(self):
        """
        Import domain information from **domain.py** in the local directory.
        No geometry is built here – we simply delegate to *domain_*.

        Example
        -------
        >>> from domain import domain_
        >>> return domain_()

        Required keys (but not limited to) in the returned dict:
        * 'mesh', 'basis'
        * 'free_dofs', 'dirichlet_dofs', 'dirichlet_value'
        """
        # TODO: from domain import domain_
        # return domain_()
        raise NotImplementedError("Define domain(...)")

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
        # TODO: from bilinear_forms import a1, a2
        # return [a1, a2]
        raise NotImplementedError("Define bilinear_forms(...)")

    def linear_forms(self):
        """
        Import element-level linear / residual forms from **linear_forms.py**.
        No assembly happens here – we just return the callables.

        Example
        -------
        >>> from linear_forms import f1, f2
        >>> return [f1, f2]
        """
        # TODO: from linear_forms import f1, f2
        # return [f1, f2]
        raise NotImplementedError("Define linear_forms(...)")

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
        # TODO: from properties import k_func, q_func
        # return [k_func, q_func]
        raise NotImplementedError("Define properties(...)")

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
        # TODO: from params import parameters
        # return parameters(n_samples)
        raise NotImplementedError("Define parameters(...)")

    # ------------------------------------------------------------------
    # Full-order model (FOM)
    # ------------------------------------------------------------------
    def fom_operators(self, cls):
        """
        Assemble (and cache) full-order operators (e.g. stiffness, mass)
        used by **fom_solver**.

        Parameters
        ----------
        cls : master_class object
            Runtime-state container injected by the master class.  Provides
            simulation metadata such as **cls.cur_itr** (current sample),
            solver tolerances, logging utilities, etc.
        """
        raise NotImplementedError("Define fom_operators(...)")

    def fom_rhs(self, cls):
        """
        Assemble (and cache) the full-order RHS vector consumed by
        **fom_solver** and hyper-reduction routines.

        Parameters
        ----------
        cls : master_class object
            Simulation context (see *fom_operators* docstring).
        """
        raise NotImplementedError("Define fom_rhs(...)")

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
        raise NotImplementedError("Define fom_solver(...)")

    # ------------------------------------------------------------------
    # Reduced-order model (ROM)
    # ------------------------------------------------------------------
    def reduced_operators(self, cls, param):
        """
        Project FOM operators onto the reduced basis so **rom_solver** can work
        in a low-dimensional space.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define reduced_operators(...)")

    def rom_solver(self, cls, param):
        """
        Solve the reduced-order model and reconstruct the high-dimensional
        field.

        Called automatically by the master class during a simulation.

        Parameters
        ----------
        cls : master_class object
            Gives access to run-time metadata (e.g. **cls.cur_itr**).
        param : ndarray or scalar
            Parameter vector/value μ.

        Returns
        -------
        u_red
            *u_red* – parameter-scaled modal coefficients  
        """
        raise NotImplementedError("Define rom_solver(...)")

    # ------------------------------------------------------------------
    # Hyper-reduction (ECSW / DEIM)
    # ------------------------------------------------------------------
    def hyper_rom_operators_ecsw(self, cls, param):
        """
        Compute operators (sampling matrices, weights, …) for the ECSW method.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define hyper_rom_operators_ecsw(...)")

    def hyper_rom_operators_deim(self, cls, param):
        """
        Compute operators (interpolation indices, projection matrices, …)
        for the DEIM method.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define hyper_rom_operators_deim(...)")

    def hyper_rom_solver_deim(self, cls, param):
        """
        Solve the DEIM hyper-reduced model.

        Called automatically by the master class when a DEIM-based simulation
        is executed.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define hyper_rom_solver_deim(...)")

    def hyper_rom_solver_ecsw(self, cls, param):
        """
        Solve the ECSW hyper-reduced model.

        Called automatically by the master class when an ECSW-based simulation
        is executed.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define hyper_rom_solver_ecsw(...)")
    
    def hyper_rom_solver_ecm(self, cls, param):
        """
        Solve the ECM hyper-reduced model.

        Called automatically by the master class when an ECM-based simulation
        is executed.

        Parameters
        ----------
        cls : master_class object   – simulation context  
        param : ndarray or scalar   – parameter vector/value μ
        """
        raise NotImplementedError("Define hyper_rom_solver_ecm(...)")