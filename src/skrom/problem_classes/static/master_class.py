"""
Static reduced-order modeling (ROM) framework.

TL;DR
-----
Runs an offline full-order FEM snapshot stage and an online reduced-order solve stage,
with optional hyper-reduction (DEIM, ECSW) to lower cost while tracking error and speed.

The module defines:
- a problem interface based on an abstract base class
- a registry for problem classes and a factory to instantiate them
- an offline workflow that runs full-order solves to generate snapshots
- an online workflow that runs ROM and computes error and timing metrics
- hooks for hyper-reduction workflows based on DEIM and ECSW

Notes
-----
Authors: Suparno Bhattacharyya; Ali Hamza Abidi Syed
"""

from pathlib import Path
import os,re,json,time,hashlib,tempfile
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
    Interface for parameterized problems used by the ROM workflow.

    The class defines the required methods for:
    - mesh and basis setup
    - affine components of bilinear and linear forms
    - parameter-dependent coefficient evaluation
    - sampling of the parameter space
    - full-order and reduced-order solvers
    - hyper-reduced solvers based on DEIM and ECSW
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
    Register a problem class under a string key.

    Parameters
    ----------
    name : str
        Registry key for the problem class.

    Returns
    -------
    deco : callable
        Decorator that adds the class to ``PROBLEM_REGISTRY`` and returns it.

    Notes
    -----
    The registry supports dynamic selection of a problem class based on the
    working directory name.
    """
    def deco(cls: Type[Problem]) -> Type[Problem]:
        PROBLEM_REGISTRY[name] = cls
        return cls
    return deco


def get_problem(name: str) -> Problem:
    """
    Instantiate a registered problem class.

    Parameters
    ----------
    name : str
        Registry key used in ``PROBLEM_REGISTRY``.

    Returns
    -------
    problem_instance : Problem
        Instance of the registered class.

    Raises
    ------
    ValueError
        If ``name`` is not present in the registry.
    """
    try:
        return PROBLEM_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown problem '{name}'. Available problems: {list(PROBLEM_REGISTRY.keys())}")


def assign_properties(prob: Problem) -> Tuple:
    """
    Collect callable handles from a problem instance.

    Parameters
    ----------
    prob : Problem
        Problem instance.

    Returns
    -------
    properties : tuple
        Tuple with:
        (parameters, bilinear_forms, linear_forms, domain, properties,
        fom_solver, rom_solver, hyper_deim_solver, hyper_ecsw_solver).
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
    Offline snapshot generation workflow.

    The workflow:
    - draws parameter samples
    - runs full-order solves
    - stores solutions and solve times
    - computes a reference field for centering
    - saves outputs to a ROM_data directory

    Attributes
    ----------
    num_snapshots : int
        Number of parameter samples.
    param_list : array_like
        Parameter samples.
    fos_solutions : list of ndarray
        Full-order solutions for each parameter sample.
    fos_time : list of float
        Solve time for each full-order solve.
    train_ref : ndarray
        Mean field computed from training snapshots.
    """

    def __init__(self, num_snapshots: int = 32):
        """
        Initialize the offline workflow.

        Parameters
        ----------
        num_snapshots : int, optional
            Number of snapshots used in the offline run. Default is 32.
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
        Run full-order solves and save offline outputs.

        Adds:
        - per-sample checkpointing to .npz
        - resume: if checkpoint exists, load it and skip the solve
        - long-path-safe filenames (Windows): hash-first + fallback to hash-only
        - crash-safe writes (atomic replace)
        - corrupt checkpoint handling (recompute + overwrite)
        """

        # reset in-memory containers (important when re-running in notebooks)
        self.fos_solutions = []
        self.fos_time = []

        cur_dir = os.getcwd()

        # keep directory name short to reduce Windows path length risk
        ckpt_dir = Path(cur_dir) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ---------- helpers ----------
        def _safe_token(s: str, maxlen: int) -> str:
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"[^0-9A-Za-z,.\-+eE_]", "_", s)
            return s[:maxlen]

        def _problem_tag() -> str:
            return (
                getattr(self, "PROBLEM_NAME", None)
                or getattr(self, "problem_name", None)
                or self.__class__.__name__
            )

        def _param_signature(param):
            """
            Returns:
            preview_token (short, human-readable),
            hash10 (stable),
            stored_param (array or string array for saving inside .npz)
            """
            try:
                arr = np.asarray(param, dtype=float).ravel()
                b = arr.tobytes()
                preview = ",".join(f"{x:.12g}" for x in arr[:6])
                if arr.size > 6:
                    preview += f",...n{arr.size}"
                stored = arr
            except Exception:
                js = json.dumps(param, sort_keys=True, default=str)
                b = js.encode("utf-8")
                preview = js
                stored = np.array([js], dtype="U")

            h10 = hashlib.sha1(b).hexdigest()[:10]
            return preview, h10, stored

        def _ckpt_path(param) -> Path:
            preview, h10, _ = _param_signature(param)
            prob = _safe_token(_problem_tag(), maxlen=20)

            # Keep preview very short; uniqueness comes from hash
            preview = _safe_token(preview, maxlen=30)

            # Option 1: slightly descriptive
            name1 = f"fos_{prob}_p{preview}_h{h10}.npz"
            path1 = ckpt_dir / name1

            # Option 2: hash-only (short path fallback)
            name2 = f"fos_{prob}_h{h10}.npz"
            path2 = ckpt_dir / name2

            # Windows long path guard
            if os.name == "nt":
                # keep headroom below 260
                if len(str(path1)) >= 240:
                    return path2

            return path1

        def _atomic_save_npz(path: Path, payload: dict):
            # IMPORTANT: temp file must end with ".npz" (np.savez appends otherwise)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent),
                prefix=path.stem + "_",
                suffix=".npz",
                delete=False,
            ) as tf:
                tmp_name = tf.name

            try:
                np.savez_compressed(tmp_name, **payload)
                os.replace(tmp_name, str(path))  # atomic on same filesystem
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

        def _save_solution(path: Path, sol, param, elapsed: float, snap_index: int):
            _, _, param_store = _param_signature(param)

            payload = {
                "snap_index": np.array([snap_index], dtype=np.int64),
                "solve_time": np.array([elapsed], dtype=float),
                "param": param_store,
            }

            if isinstance(sol, tuple):
                payload["nsol"] = np.array([len(sol)], dtype=np.int32)
                for j, x in enumerate(sol):
                    payload[f"sol{j}"] = np.asarray(x)
            else:
                payload["nsol"] = np.array([1], dtype=np.int32)
                payload["sol0"] = np.asarray(sol)

            _atomic_save_npz(path, payload)

        def _load_solution(path: Path):
            with np.load(path, allow_pickle=False) as data:
                nsol = int(data["nsol"][0]) if "nsol" in data.files else 1
                t = float(data["solve_time"][0]) if "solve_time" in data.files else np.nan
                if nsol == 1:
                    sol = data["sol0"]
                else:
                    sol = tuple(data[f"sol{j}"] for j in range(nsol))
            return sol, t

        # ---------- main loop ----------
        for i, param in enumerate(self.param_list):
            print(f"Snap {i+1}/{len(self.param_list)} params={param}")
            self.cur_itr = i

            path = _ckpt_path(param)

            if path.exists():
                try:
                    sol, t_saved = _load_solution(path)
                    self.fos_time.append(t_saved)
                except Exception:
                    # corrupt/partial file -> delete and recompute
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    t0 = time.perf_counter()
                    sol = self.fom_solver(cls=self, param=param)
                    elapsed = time.perf_counter() - t0
                    self.fos_time.append(elapsed)
                    _save_solution(path, sol, param, elapsed, snap_index=i)
            else:
                t0 = time.perf_counter()
                sol = self.fom_solver(cls=self, param=param)
                elapsed = time.perf_counter() - t0
                self.fos_time.append(elapsed)
                _save_solution(path, sol, param, elapsed, snap_index=i)

            # store solution copy (in-memory)
            if isinstance(sol, tuple):
                self.fos_solutions.append(tuple(np.copy(x) for x in sol))
            else:
                self.fos_solutions.append(np.copy(sol))

        # ---------- convert + mean over training subset ----------
        if len(self.fos_solutions) == 0:
            return

        first = self.fos_solutions[0]
        if isinstance(first, tuple):
            self.fos_solutions = np.array(self.fos_solutions, dtype=object)

            train_ids = np.where(self.train_mask)[0]
            ncomp = len(first)
            train_ref = []
            for j in range(ncomp):
                stack_j = np.stack([self.fos_solutions[k][j] for k in train_ids], axis=0)
                train_ref.append(np.mean(stack_j, axis=0))
            self.train_ref = tuple(train_ref)
        else:
            self.fos_solutions = np.asarray(self.fos_solutions)
            self.train_ref = np.mean(self.fos_solutions[self.train_mask], axis=0)

        # ---------- Persist new attributes to disk ----------
        new = set(vars(self)) - self._baseline_attrs
        save_dict = {k: getattr(self, k) for k in new if not k.startswith("_")}
        rom_data_gen(save_dict, cur_dir)



# ─────────────────────────────────────────────────────────────
# ONLINE ROM EVALUATION
# ─────────────────────────────────────────────────────────────
class rom_simulation:
    """
    Online ROM evaluation workflow.

    The workflow:
    - loads ROM_data outputs from disk
    - selects test parameters and reference full-order solutions
    - runs ROM solvers and reconstructs full fields
    - computes relative error and time ratio metrics
    - supports hyper-reduced solvers based on DEIM and ECSW

    Attributes
    ----------
    V_sel : ndarray
        Basis matrix used for reconstruction.
    n_sel : int
        Mode count used by the ROM.
    param_list_test : array_like
        Test parameter samples.
    rom_error : list of float
        Relative error values in percent for each test case.
    speed_up : list of float
        Time ratios computed as (full-order time) / (ROM time).
    """

    def __init__(
        self, train_ref=None, test_ref=None, fos_solutions=None,
        train_mask=None, test_mask=None,
        V_sel=None, n_sel=None, N_rom_snap=None
    ):
        """
        Initialize the online workflow and load ROM_data outputs.

        Parameters
        ----------
        train_ref : ndarray, optional
            Reference field used for centering during training.
        test_ref : ndarray, optional
            Reference field used for reconstruction on the test set.
        fos_solutions : ndarray, optional
            Full-order snapshots. If None, data are loaded from disk.
        train_mask : array_like of bool, optional
            Training mask. If None, data are loaded from disk.
        test_mask : array_like of bool, optional
            Test mask. If None, data are loaded from disk.
        V_sel : ndarray, optional
            Basis matrix used in reconstruction.
        n_sel : int, optional
            Mode count used by the ROM.
        N_rom_snap : int, optional
            Test case count evaluated in the run. If None, uses all test cases.

        Notes
        -----
        The initializer loads ``ROM_data`` from the current working directory.
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

        if train_ref is not None:
            self.train_ref = train_ref

        if test_ref is not None:
            self.test_ref = test_ref
        else:
            self.test_ref = self.train_ref

        if fos_solutions is not None:
            self.fos_solutions = fos_solutions

        if train_mask is not None:
            self.train_mask = train_mask
        
        if test_mask is not None:
            self.test_mask = test_mask

        # Prepare test/training splits
        self.param_list_test = self.param_list[self.test_mask]
        self.fos_test_data   = self.fos_solutions[self.test_mask]
        self.fos_test_time   = np.asarray(self.fos_time)[self.test_mask]
        self.sol_train_ms    = self.fos_solutions[self.train_mask] - self.train_ref

        # Store basis info
        self.V_sel      = V_sel
        self.n_sel      = n_sel
        self.N_rom_snap = N_rom_snap or len(self.param_list_test)

        # Ensure attributes are unwrapped if zero-dimensional
        unwrap_attr(self, 'basis')
        unwrap_attr(self, 'mesh')

    def run_rom_simulation(self):
        """
        Run ROM evaluation on the test set.

        The method:
        - runs the reduced solver for each test parameter
        - reconstructs a full field from reduced coordinates
        - computes relative error in percent
        - computes time ratio (full-order time) / (ROM time)

        Returns
        -------
        rom_error : list of float
            Relative error values in percent.
        speed_up : list of float
            Time ratios for each test case.
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

            if self.test_ref is not None:

                if len(self.test_ref.shape)==3:
                    sol_rom = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i])
                else:
                    sol_rom = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)

            dt = time.perf_counter() - t0
            # Compute error & speed-up
            sol_fos = self.fos_test_data[i]

            if sol_rom.shape != sol_fos.shape and sol_rom.ndim == 2 and sol_rom.T.shape == sol_fos.shape:
                sol_rom = sol_rom.T

            if sol_rom.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_rom.shape}")
            
            err   = 100 * np.linalg.norm(sol_fos - sol_rom) \
                    / np.linalg.norm(sol_fos)
            speed = self.fos_test_time[i] / dt

            self.rom_error.append(err)
            self.speed_up.append(speed)
            self.rom_solutions.append(sol_rom.copy())

        return self.rom_error, self.speed_up
    
    def run_hyper_rom_simulation_ecsw(self, z):
        """
        Run ECSW hyper-ROM evaluation on the test set.

        Parameters
        ----------
        z : array_like
            Element weight vector used by the ECSW solver.

        Returns
        -------
        hyper_rom_error : list of float
            Relative error values in percent.
        hyper_speed_up : list of float
            Time ratios for each test case.
        """
        self.hyper_speed_up    = []
        self.hyper_rom_error   = []
        self.hyper_rom_solutions = []
        # Store hyper-reduction parameters
        self.z            = z

        for i, param in enumerate(self.param_list_test[:self.N_rom_snap]):
            print(f"Snap {i+1}/{len(self.param_list_test)} params={param}")
            self.cur_itr = i

            # Time the hyper-ROM solve + reconstruction
            t0 = time.perf_counter()
            sol_red_ = self.hyper_rom_solver_ecsw(cls=self, param=param)

            if len(self.test_ref.shape)==3:
                sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i])
            else:
                sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)


            dt = time.perf_counter() - t0

            # record speed-up and solution
            self.hyper_speed_up.append(self.fos_test_time[i] / dt)

            # compute and record error
            sol_fos = self.fos_test_data[i]

            if sol_hyper.shape != sol_fos.shape and sol_hyper.ndim == 2 and sol_hyper.T.shape == sol_fos.shape:
                sol_hyper = sol_hyper.T

            if sol_hyper.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyper.shape}")

            self.hyper_rom_solutions.append(sol_hyper.copy())

            err = 100 * np.linalg.norm(sol_fos - sol_hyper) / np.linalg.norm(sol_fos)
            self.hyper_rom_error.append(err)

        return self.hyper_rom_error, self.hyper_speed_up


    def run_hyper_rom_simulation_deim(self, z, deim_mat, sampled_rows):
        """
        Run DEIM hyper-ROM evaluation on the test set.

        Parameters
        ----------
        z : array_like
            Weight vector stored on the instance.
        deim_mat : ndarray
            Interpolation matrix used by the DEIM workflow.
        sampled_rows : array_like of int
            Indices of sampled degrees of freedom.

        Returns
        -------
        hyper_rom_error : list of float
            Relative error values in percent.
        hyper_speed_up : list of float
            Time ratios for each test case.
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

            if len(self.test_ref.shape)==3:
                sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i])
            else:
                sol_hyper = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)

            dt = time.perf_counter() - t0

            # record speed-up and solution
            self.hyper_speed_up.append(self.fos_test_time[i] / dt)
            self.hyper_rom_solutions.append(sol_hyper.copy())

            # compute and record error
            sol_fos = self.fos_test_data[i]

            if sol_hyper.shape != sol_fos.shape and sol_hyper.ndim == 2 and sol_hyper.T.shape == sol_fos.shape:
                sol_hyper = sol_hyper.T

            if sol_hyper.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyper.shape}")
            
            
            err = 100 * np.linalg.norm(sol_fos - sol_hyper) / np.linalg.norm(sol_fos)
            self.hyper_rom_error.append(err)

        return self.hyper_rom_error, self.hyper_speed_up
