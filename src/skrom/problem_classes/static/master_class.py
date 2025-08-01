"""
Static reduced-order modeling (ROM) framework for efficient numerical simulation.

This module provides a complete framework for reduced-order modeling including:
- Dynamic import of problem definitions via abstract base classes
- Full-order FEM solver (FOM) for generating training snapshots
- Offline snapshot generator for building reduced bases
- Online ROM evaluator with performance metrics
- Support for hyper-reduction techniques (DEIM, ECSW)

**TL;DR**: Enables fast approximation of expensive PDE solutions by learning 
from precomputed snapshots, achieving significant computational speedups while 
maintaining acceptable accuracy.

Authors: Suparno Bhattacharyya, Ali Hamza Abidi Syed
"""

from pathlib import Path
import os
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type
from skrom.fom.fem_utils import unwrap_attr
from skrom.rom.rom_utils import rom_data_gen, load_rom_data, reconstruct_solution


# ─────────────────────────────────────────────────────────────
# PROBLEM DETECTION
# ─────────────────────────────────────────────────────────────
cwd = Path.cwd()
# If running inside a folder named "problem_X", use that; else raise an error
if cwd.name.startswith("problem_"):
    PROBLEM = cwd.name
else:
    raise ValueError(
        "Current directory must be named 'problem_X' where X can be a number/string.")


# ─────────────────────────────────────────────────────────────
# PROBLEM REGISTRY AND INTERFACE
# ─────────────────────────────────────────────────────────────
class Problem(ABC):
    """
    Abstract base class for parameterized PDE problems with affine decomposition.

    **TL;DR**: Defines the interface that all ROM problems must implement,
    ensuring consistent structure for domain setup, assembly, and solving.

    This abstract class serves as a blueprint for defining parameterized partial 
    differential equation problems suitable for reduced-order modeling. 
    """
    @abstractmethod
    def domain(self):
        """Return geometry and FEM bases."""
        pass

    @abstractmethod
    def bilinear_forms(self):
        """Return affine bilinear form components."""
        pass

    @abstractmethod
    def linear_forms(self):
        """Return affine linear form components."""
        pass

    @abstractmethod
    def properties(self):
        """Return function to compute parameter-dependent coefficients."""
        pass

    @abstractmethod
    def parameters(self):
        """Generate sampling of parameter space."""
        pass

    @abstractmethod
    def fom_solver(self):
        """Solve full-order FEM system for given parameters."""
        pass

    @abstractmethod
    def rom_solver(self):
        """Solve reduced-order model for given parameters."""
        pass

    @abstractmethod
    def hyper_rom_solver_deim(self):
        """Solve DEIM-based hyper-reduced system for given parameters."""
        pass

    @abstractmethod
    def hyper_rom_solver_ecsw(self):
        """Solve ECSW-based hyper-reduced system for given parameters."""
        pass


# Decorator to register problem classes by name
PROBLEM_REGISTRY: Dict[str, Type[Problem]] = {}


def register_problem(name: str):
    """
    Decorator to register a problem class in the global registry.

    **TL;DR**: Allows automatic discovery and instantiation of problem classes
    by name, enabling modular problem definitions.

    Parameters
    ----------
    name : str
        Unique identifier for the problem class.

    Returns
    -------
    decorator : callable
        Decorator function that registers the class.
    """
    def deco(cls: Type[Problem]) -> Type[Problem]:
        PROBLEM_REGISTRY[name] = cls
        return cls
    return deco


def get_problem(name: str) -> Problem:
    """
    Instantiate a registered problem class by name.

    **TL;DR**: Factory function to create problem instances from the registry
    using string identifiers.

    Parameters
    ----------
    name : str
        Name of the registered problem class.

    Returns
    -------
    problem_instance : Problem
        Instantiated problem object.

    Raises
    ------
    ValueError
        If the requested problem name is not found in the registry.
    """
    try:
        return PROBLEM_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown problem '{name}'. Available problems: {list(PROBLEM_REGISTRY.keys())}")


def assign_properties(prob: Problem) -> Tuple:
    """
    Extract and organize all problem methods and properties.

    **TL;DR**: Convenience function to unpack all problem components into 
    a structured tuple for easy access.

    Parameters
    ----------
    prob : Problem
        Problem instance to extract properties from.

    Returns
    -------
    properties : tuple
        Tuple containing (parameters, bilinear_forms, linear_forms, domain,
        properties, fom_solver, rom_solver, hyper_deim_solver, hyper_ecsw_solver).

    """
    parameters        = prob.parameters
    a                 = prob.bilinear_forms()
    l                 = prob.linear_forms()
    domain_           = prob.domain
    properties        = prob.properties()
    fom_solver        = prob.fom_solver
    rom_solver        = prob.rom_solver
    hyper_deim_solver = prob.hyper_rom_solver_deim
    hyper_ecsw_solver = prob.hyper_rom_solver_ecsw

    return (
        parameters, a, l, domain_, properties,
        fom_solver,
        rom_solver,
        hyper_deim_solver,
        hyper_ecsw_solver,
    )


# ─────────────────────────────────────────────────────────────
# OFFLINE SNAPSHOT GENERATION
# ─────────────────────────────────────────────────────────────
class fom_simulation:
    """
    Offline snapshot generator using full-order finite element method.

    **TL;DR**: Generates training data by solving the full-order system at 
    multiple parameter values, storing solutions and timing information for 
    later ROM construction.

    This class orchestrates the offline phase of ROM construction by:
    1. Sampling parameter space
    2. Solving full-order systems at each parameter
    3. Recording solutions and computational times
    4. Computing mean solution for centering
    5. Saving all data for ROM construction

    The generated snapshots form the columns of the snapshot matrix used 
    to build the reduced basis via proper orthogonal decomposition (POD) 
    or other dimensionality reduction techniques.

    Attributes
    ----------
    num_snapshots : int
        Number of parameter samples/snapshots to generate.
    param_list : array_like
        Parameter vectors for snapshot generation.
    fos_solutions : list of ndarray
        Full-order solutions at each parameter value.
    fos_time : list of float
        Solution times for each full-order solve.
    mean : ndarray
        Mean solution used for centering the snapshot matrix.

    Examples
    --------
    >>> sim = fom_simulation(num_snapshots=50)
    >>> sim.run_simulation()
    >>> # Solutions and timing data saved to ROM_data/ directory
    """

    def __init__(self, num_snapshots: int = 32):
        """
        Initialize simulation parameters and bind problem methods.

        Parameters
        ----------
        num_snapshots : int, default=32
            Number of snapshots to generate for ROM training. More snapshots
            generally improve ROM accuracy but increase offline computational cost.
        """
        # Bind prob methods
        prob = get_problem(PROBLEM)
        (
        self.parameters,
        self.bilinear_forms,
        self.linear_forms,
        self.domain,
        self.properties,
        self.fom_solver,
        *_
        ) = assign_properties(prob)

        # Track attributes introduced by this class
        self._baseline_attrs = set(vars(self))

        # Load domain and basis
        data = self.domain()
        self.mesh  = data["mesh"]
        self.basis = data["basis"]

        # Prepare parameter sampling
        self.num_snapshots = num_snapshots
        (
            self.param_list,
            self.param_range,
            self.train_mask,
            self.test_mask
        ) = self.parameters(num_snapshots)

        # Containers for solutions and timings
        self.fos_solutions = []
        self.fos_time      = []

    def run_simulation(self) -> None:
        """
        Execute snapshot generation and save results.

        **TL;DR**: Main execution method that solves FOM at all parameter values,
        records timings, and saves data required for ROM derivation to disk.

        This method performs the complete offline phase:
        1. Iterates through all parameter values
        2. Solves full-order system at each parameter
        3. Records solution time and stores solution
        4. Computes mean solution over training set
        5. Saves all new attributes to ROM_data directory

        The timing information is crucial for computing speedup metrics
        during online ROM evaluation.

        Notes
        -----
        Solutions are deep-copied to avoid memory aliasing issues.
        Only attributes created after initialization are saved to disk
        to avoid redundant storage of problem methods.
        """
        for i, param in enumerate(self.param_list):
            print(f"Snap {i+1}/{len(self.param_list)} params={param}")
            self.cur_itr = i

            # Time the FOM solve
            t0 = time.perf_counter()
            sol = self.fom_solver(cls=self, param=param)
            self.fos_time.append(time.perf_counter() - t0)

            # Store solution copy
            if isinstance(sol, tuple):
                self.fos_solutions.append(tuple(np.copy(x) for x in sol))
            else:
                self.fos_solutions.append(np.copy(sol))

        # Convert to array and compute mean over training set
        self.fos_solutions = np.array(self.fos_solutions)
        self.mean = np.mean(self.fos_solutions[self.train_mask], axis=0)

        # Persist new attributes to disk
        new = set(vars(self)) - self._baseline_attrs
        save_dict = {
            k: getattr(self, k)
            for k in new if not k.startswith("_")
        }
        cur_dir = os.getcwd()
        rom_data_gen(save_dict, cur_dir)


# ─────────────────────────────────────────────────────────────
# ONLINE ROM EVALUATION
# ─────────────────────────────────────────────────────────────
class rom_simulation:
    """
    Online ROM evaluator with error analysis and performance metrics.

    **TL;DR**: Evaluates ROM performance by comparing reduced-order solutions 
    against full-order references, computing error percentages and computational 
    speedups.

    This class handles the online phase of ROM evaluation by:
    1. Loading precomputed ROM data (snapshots, basis, parameters)
    2. Solving reduced-order systems at test parameters
    3. Reconstructing full-order solutions from ROM coefficients
    4. Computing error metrics and speedup ratios
    5. Supporting both standard Galerkin ROM and hyper-reduced variants

    The class provides comprehensive performance analysis including relative 
    errors and computational speedups, essential for validating ROM accuracy 
    and efficiency.

    Attributes
    ----------
    V_sel : ndarray
        Reduced basis matrix of shape (n_dofs, n_modes).
    n_sel : int
        Number of ROM modes/basis functions.
    param_list_test : array_like
        Test parameter values for ROM evaluation.
    rom_error : list of float
        Relative error percentages for each test case.
    speed_up : list of float
        Computational speedup ratios (FOM_time / ROM_time).
    """

    def __init__(
        self, mean=None, fos_solutions=None,
        train_mask=None, test_mask=None,
        V_sel=None, n_sel=None, N_rom_snap=None
    ):
        """
        Initialize ROM simulation with data and basis information.

        Parameters
        ----------
        mean : ndarray, optional
            Mean snapshot for solution centering. If None, loaded from disk.
        fos_solutions : ndarray, optional
            Full-order snapshot matrix. If None, loaded from disk.
        train_mask : array_like of bool, optional
            Training parameter mask. If None, loaded from disk.
        test_mask : array_like of bool, optional
            Test parameter mask. If None, loaded from disk.
        V_sel : ndarray, optional
            Reduced basis matrix of shape (n_dofs, n_modes).
        n_sel : int, optional
            Number of ROM modes to use.
        N_rom_snap : int, optional
            Number of ROM test cases to run. If None, uses all test parameters.

        Notes
        -----
        If data parameters are None, they will be loaded from the ROM_data
        directory. The V_sel basis matrix is typically computed via POD
        on the mean-centered snapshot matrix.
        """
        # Bind prob methods
        prob = get_problem(PROBLEM)
        (
            self.parameters,
            self.bilinear_forms,
            self.linear_forms,
            self.domain,
            self.properties,
            _,
            self.rom_solver,
            self.hyper_rom_solver_deim,
            self.hyper_rom_solver_ecsw,
        ) = assign_properties(prob)

        # Load ROM data from disk
        cur_dir = os.getcwd()
        rom_dir = os.path.join(cur_dir, "ROM_data")
        load_rom_data(self, rom_dir)

        # Prepare test/training splits
        self.param_list_test = self.param_list[self.test_mask]
        self.fos_test_data   = self.fos_solutions[self.test_mask]
        self.fos_test_time   = np.asarray(self.fos_time)[self.test_mask]
        self.sol_train_ms    = self.fos_solutions[self.train_mask] - self.mean

        # Store basis info
        self.V_sel      = V_sel
        self.n_sel      = n_sel
        self.N_rom_snap = N_rom_snap or len(self.param_list_test)

        # Ensure attributes are unwrapped if zero-dimensional
        unwrap_attr(self, 'basis')
        unwrap_attr(self, 'mesh')

    def run_rom_simulation(self):
        """
        Execute ROM evaluation and compute performance metrics.

        **TL;DR**: Runs ROM at test parameters, reconstructs solutions, and 
        computes error percentages and speedup ratios versus full-order model.

        This method performs the complete ROM evaluation:
        1. Iterates through test parameter values
        2. Solves reduced-order system (much faster than FOM)
        3. Reconstructs full-order solution: u_ROM = V * u_reduced + mean
        4. Computes relative error: ||u_FOM - u_ROM|| / ||u_FOM|| * 100%
        5. Computes speedup ratio: t_FOM / t_ROM

        Returns
        -------
        rom_error : list of float
            Relative error percentages for each test parameter.
        speed_up : list of float
            Computational speedup ratios for each test parameter.
        """
        self.speed_up     = []
        self.rom_error    = []
        self.rom_solutions = []

        for i, param in enumerate(self.param_list_test[:self.N_rom_snap]):
            print(f"Snap {i+1}/{len(self.param_list)} params={param}")
            self.cur_itr = i

            # Time the ROM solve
            t0 = time.perf_counter()
            sol_red_ = self.rom_solver(cls=self, param=param)
            sol_rom = reconstruct_solution(sol_red_, self.V_sel, self.mean)
            dt = time.perf_counter() - t0

            # Compute error & speed-up
            sol_fos = self.fos_test_data[i]
            err   = 100 * np.linalg.norm(sol_fos - sol_rom) \
                    / np.linalg.norm(sol_fos)
            speed = self.fos_test_time[i] / dt

            self.rom_error.append(err)
            self.speed_up.append(speed)
            self.rom_solutions.append(sol_rom.copy())

        return self.rom_error, self.speed_up
    
    def run_hyper_rom_simulation_ecsw(self, z):
        """
        Execute ECSW hyper-ROM evaluation with performance analysis.

        **TL;DR**: Runs Element-based Empirical Cubature in Strongly Weighted 
        (ECSW) hyper-reduced model, achieving further speedups over standard ROM
        by reducing integration costs.

        ECSW hyper-reduction reduces computational cost by integrating the 
        weak form only over a weighted subset of elements, rather than the 
        full domain. This is particularly effective for problems where the 
        solution has localized features.

        Parameters
        ----------
        z : array_like
            Element weight vector for ECSW hyper-reduction. Elements with 
            larger weights contribute more to the reduced integration.

        Returns
        -------
        hyper_rom_error : list of float
            Relative error percentages versus full-order solutions.
        hyper_speed_up : list of float
            Computational speedup ratios versus full-order model.
        """
        self.hyper_speed_up    = []
        self.hyper_rom_error   = []
        self.hyper_rom_solutions = []
        # Store hyper-reduction parameters
        self.z            = z

        for i, param in enumerate(self.param_list_test[:self.N_rom_snap]):
            print(f"Snap {i+1}/{len(self.param_list)} params={param}")
            self.cur_itr = i

            # Time the hyper-ROM solve + reconstruction
            t0 = time.perf_counter()
            sol_red_ = self.hyper_rom_solver_ecsw(cls=self, param=param)
            sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.mean)
            dt = time.perf_counter() - t0

            # record speed-up and solution
            self.hyper_speed_up.append(self.fos_test_time[i] / dt)
            self.hyper_rom_solutions.append(sol_hyper.copy())

            # compute and record error
            sol_fos = self.fos_test_data[i]
            err = 100 * np.linalg.norm(sol_fos - sol_hyper) / np.linalg.norm(sol_fos)
            self.hyper_rom_error.append(err)

        return self.hyper_rom_error, self.hyper_speed_up

    def run_hyper_rom_simulation_deim(self, z, deim_mat, sampled_rows):
        """
        Execute DEIM hyper-ROM evaluation with performance analysis.

        **TL;DR**: Runs Discrete Empirical Interpolation Method (DEIM) 
        hyper-reduction for efficient handling of nonlinear terms by 
        interpolation at selected points.

        Parameters
        ----------
        z : array_like
            Weight vector for hyper-reduction stored for reference.
        deim_mat : ndarray
            DEIM interpolation matrix computed offline.
        sampled_rows : array_like of int
            Indices of degrees of freedom used as DEIM interpolation points.

        Returns
        -------
        hyper_rom_error : list of float
            Relative error percentages versus full-order solutions.
        hyper_speed_up : list of float
            Computational speedup ratios versus full-order model.
        """
        self.hyper_speed_up    = []
        self.hyper_rom_error   = []
        self.hyper_rom_solutions = []
        # Store hyper-reduction parameters
        self.z            = z
        self.deim_mat   = deim_mat
        self.sampled_rows = sampled_rows

        for i, param in enumerate(self.param_list_test[:self.N_rom_snap]):
            print(f"Snap {i+1}/{len(self.param_list)} params={param}")
            self.cur_itr = i

            # Time the hyper-ROM solve + reconstruction
            t0 = time.perf_counter()
            sol_red_ = self.hyper_rom_solver_deim(cls=self, param=param)
            sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.mean)
            dt = time.perf_counter() - t0

            # record speed-up and solution
            self.hyper_speed_up.append(self.fos_test_time[i] / dt)
            self.hyper_rom_solutions.append(sol_hyper.copy())

            # compute and record error
            sol_fos = self.fos_test_data[i]
            err = 100 * np.linalg.norm(sol_fos - sol_hyper) / np.linalg.norm(sol_fos)
            self.hyper_rom_error.append(err)

        return self.hyper_rom_error, self.hyper_speed_up
