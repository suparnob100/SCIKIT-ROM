"""
Module for static reduced-order modeling (ROM):

- Dynamic import of problem definitions
- Full-order FEM solver (FOM)
- Offline snapshot generator
- Online ROM evaluator
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
    Abstract base for conductivity problems under affine decomposition.
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
    Decorator to register a problem class under a given name.
    """
    def deco(cls: Type[Problem]) -> Type[Problem]:
        PROBLEM_REGISTRY[name] = cls
        return cls
    return deco


def get_problem(name: str) -> Problem:
    """
    Instantiate a registered problem by name.
    """
    try:
        return PROBLEM_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown problem '{name}'")



def assign_properties(prob: Problem) -> Tuple:
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
    Offline snapshot generator using full-order FEM.
    """
    def __init__(self, num_snapshots: int = 32):
        """
        Initialize simulation parameters and storage.

        Parameters
        ----------
        num_snapshots : int
            Number of snapshots to generate.
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
        Generate snapshots and record timings.
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
    Plain Galerkin ROM evaluator with error and speed-up metrics.
    """
    def __init__(
        self, mean=None, fos_solutions=None,
        train_mask=None, test_mask=None,
        V_sel=None, n_sel=None, N_rom_snap=None
    ):
        """
        Load ROM data and prepare test/training splits.

        Parameters
        ----------
        mean : ndarray
            Mean snapshot.
        fos_solutions : ndarray
            Full-order snapshot matrix.
        train_mask : array_like
            Boolean mask for training.
        test_mask : array_like
            Boolean mask for testing.
        V_sel : ndarray
            Reduced basis.
        n_sel : int
            Number of modes.
        N_rom_snap : int, optional
            # of ROM snapshots to run.
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
        Execute ROM solves, compute error percentages and speed-ups.

        Returns
        -------
        rom_error : list of float
            Percent error per snapshot.
        speed_up : list of float
            Full/ROM time ratio per snapshot.
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
        Execute hyper-ROM solves, compute error percentages and speed-ups.

        Parameters
        ----------
        z : array_like
            Weight vector for hyper-reduction (stored for reference).

        Returns
        -------
        hyper_rom_error : list of float
            Percent error per snapshot.
        hyper_speed_up : list of float
            Full/FOM time ratio per snapshot.
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
        Execute hyper-ROM solves, compute error percentages and speed-ups.

        Parameters
        ----------
        z : array_like
            Weight vector for hyper-reduction (stored for reference).

        Returns
        -------
        hyper_rom_error : list of float
            Percent error per snapshot.
        hyper_speed_up : list of float
            Full/FOM time ratio per snapshot.
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
    