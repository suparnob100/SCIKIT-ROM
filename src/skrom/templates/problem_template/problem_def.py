import numpy as np
import os
from skrom.problem_classes.static.master_class import register_problem, Problem
# derive the problem name from this file’s parent folder
PROBLEM_NAME = os.path.basename(os.path.dirname(__file__))

@register_problem(PROBLEM_NAME)
class ProblemTemplate(Problem):
    """
    Template for an affine or non-linear ROM problem.
    """

    def domain(self):
        """
        Setup mesh, basis, DOFs, and boundary conditions.

        Returns:
            dict with keys like
            - 'mesh'
            - 'basis'
            - 'free_dofs'
            - 'dirichlet_dofs'
            - 'dirichlet_value'
            # …any other domain-specific items…
        """
        # TODO: from domain import domain_ 
        # return domain_()
        raise NotImplementedError("Define domain(...)")

    def bilinear_forms(self):
        """
        Return a list of bilinear form callables (or Jacobian forms for nonlinear problems).
        Each form should have signature a(u, v, w).
        """
        # TODO: from bilinear_forms import a
        # return [a1, a2, …]
        raise NotImplementedError("Define bilinear_forms(...)")

    def linear_forms(self):
        """
        Return a list of linear form callables (or residual forms for nonlinear problems).
        Each form should have signature l(v, w).
        """
        # TODO: from linear_forms import l 
        # return [l1, l2, …]
        raise NotImplementedError("Define linear_forms(...)")

    def properties(self):
        """
        Return a list of functions mapping physical parameters to coefficients.
        E.g. [k_func, q_func, …]
        """
        # TODO: from properties import properties_fn
        # return [prop_1, prop_2, …]
        raise NotImplementedError("Define properties(...)")

    def parameters(self, n_samples):
        """
        Generate sampling design over parameter space.

        Parameters:
            n_samples (int): number of samples to draw.

        Returns:
            params, param_ranges, train_mask, test_mask
        """
        # TODO: from params import parameters
        # return parameters(n_samples)
        raise NotImplementedError("Define parameters(...)")

    def fom_operators(self, cls):
        """
        Assemble (and cache on first call) the full-order operators
        needed by the FOM solver (e.g. stiffness matrix).

        Returns:
            tuple of FOM operators
        """
        # TODO: assemble & cache FOM operators
        raise NotImplementedError("Define fom_operators(...)")
    
    def fom_rhs(self, cls):
        """
        Assemble (and cache on first call) the full-order right-hand side vector.

        Returns:
            full-order RHS vector
        """
        # TODO: implement ROM solve
        raise NotImplementedError("Define fom_rhs(...)")

    def fom_solver(self, cls, param):
        """
        Solve the full-order model for given parameters.

        Returns:
            solution array satisfying BCs
        """
        # TODO: implement FOM solve
        raise NotImplementedError("Define fom_solver(...)")

    def reduced_operators(self, cls, param):
        """
        Project FOM operators onto the reduced basis and compute
        any mean or offset terms needed for the ROM.

        Returns:
            (modal_solution_full_space, mean_correction)
        """
        # TODO: implement ROM projection
        raise NotImplementedError("Define reduced_operators(...)")

    def rom_solver(self, cls, param):
        """
        Solve the reduced-order model and reconstruct the full solution.

        Returns:
            (parameter_scaled_modal_solution, mean_correction)
        """
        # TODO: implement ROM solve
        raise NotImplementedError("Define rom_solver(...)")
    

    
    def hyper_rom_solver(self):
        """Solve hyper-reduced-order model for given parameters."""
    pass