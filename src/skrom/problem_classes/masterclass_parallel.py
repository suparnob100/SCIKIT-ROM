"""
Parallel implementation of master_class using multithreading for snapshot generation and ROM evaluation.

Notes
-----
Authors: Suparno Bhattacharyya
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import time
import logging
import contextvars
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import importlib
import importlib.util
import inspect
import traceback
import numpy as np

from skrom.fom.fem_utils import unwrap_attr
from skrom.rom.rom_utils import rom_data_gen, load_rom_data, reconstruct_solution


# ─────────────────────────────────────────────────────────────
# WINDOWS-SAFE FILE HELPERS
# ─────────────────────────────────────────────────────────────
def _win_long_path(path):
    """
    Return a Windows extended-length path when running on Windows.

    This avoids FileNotFoundError/OSError caused by long OneDrive/project paths
    during checkpoint writes. On non-Windows systems, it returns the normal
    absolute path string.
    """
    path_str = os.path.abspath(os.fspath(path))
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str[2:]
    return "\\\\?\\" + path_str


# ─────────────────────────────────────────────────────────────
# BASIC LOGGER (optional)
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PROBLEM DETECTION
# ─────────────────────────────────────────────────────────────
cwd = Path.cwd()

PROBLEM = os.environ.get("SKROM_PROBLEM", "").strip()
if not PROBLEM:
    if cwd.name.startswith("problem_"):
        PROBLEM = cwd.name
    else:
        raise ValueError(
            "Current directory must be named 'problem_X' where X can be a number/string, "
            "or set env var SKROM_PROBLEM=problem_X."
        )


def _detect_problem_dir(problem_name: str) -> Path:
    """
    Locate the active problem directory.

    The usual workflow is to run from inside a folder named ``problem_*``.
    For script-based workflows, ``SKROM_PROBLEM_DIR`` may point directly to the
    folder containing ``problem_def.py``.
    """
    env_dir = os.environ.get("SKROM_PROBLEM_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"SKROM_PROBLEM_DIR points to a missing directory: {p}")
        return p

    if cwd.name == problem_name and (cwd / "problem_def.py").exists():
        return cwd

    if cwd.name.startswith("problem_") and (cwd / "problem_def.py").exists():
        return cwd

    if (cwd / problem_name / "problem_def.py").exists():
        return cwd / problem_name

    # Fallback: keep cwd to produce an informative error below.
    return cwd


PROBLEM_DIR = _detect_problem_dir(PROBLEM)


# ─────────────────────────────────────────────────────────────
# PROBLEM REGISTRY AND INTERFACE
# ─────────────────────────────────────────────────────────────
class Problem(ABC):
    @abstractmethod
    def domain(self): ...
    @abstractmethod
    def bilinear_forms(self): ...
    @abstractmethod
    def linear_forms(self): ...
    @abstractmethod
    def properties(self): ...
    @abstractmethod
    def parameters(self): ...
    @abstractmethod
    def fom_solver(self): ...
    @abstractmethod
    def rom_solver(self): ...
    @abstractmethod
    def hyper_rom_solver_deim(self): ...
    @abstractmethod
    def hyper_rom_solver_ecsw(self): ...
    @abstractmethod
    def hyper_rom_solver_ecm(self): ...


PROBLEM_REGISTRY: Dict[str, Type[Problem]] = {}


def _module_aliases() -> Tuple[str, ...]:
    """
    Common names by which problem_def.py may import the master class module.

    Aliasing these names to this module prevents separate registry dictionaries
    when ``problem_def.py`` imports ``register_problem`` using a different path.
    """
    return (
        "skrom.problem_classes.masterclass_parallel",
        "masterclass_parallel",
        "skrom.problem_classes.masterclass",
        "masterclass",
        "skrom.problem_classes.static.master_class",
        "skrom.problem_classes.static.masterclass",
    )


def _unify_masterclass_module_aliases() -> None:
    current_module = sys.modules[__name__]
    for alias in _module_aliases():
        if sys.modules.get(alias) is not current_module:
            sys.modules[alias] = current_module


def register_problem(name: str):
    """
    Register a problem class under a string key.
    """
    def deco(cls: Type[Problem]) -> Type[Problem]:
        PROBLEM_REGISTRY[name] = cls
        return cls
    return deco


def _add_sys_path(p: Path) -> None:
    sp = str(p)
    if sp and sp not in sys.path:
        sys.path.insert(0, sp)


def _merge_registry_from_module_name(modname: str) -> None:
    mod = sys.modules.get(modname)
    if mod is None:
        return
    reg = getattr(mod, "PROBLEM_REGISTRY", None)
    if isinstance(reg, dict) and reg:
        PROBLEM_REGISTRY.update(reg)


def _merge_known_registries() -> None:
    for modname in _module_aliases():
        _merge_registry_from_module_name(modname)


def _auto_register_problem_from_module(module, name: str) -> None:
    """
    Fallback registration for a local ``problem_def.py`` that defines one
    concrete problem class but does not explicitly use ``@register_problem``.
    """
    if name in PROBLEM_REGISTRY:
        return

    required_methods = (
        "domain",
        "bilinear_forms",
        "linear_forms",
        "properties",
        "parameters",
        "fom_solver",
        "rom_solver",
        "hyper_rom_solver_deim",
        "hyper_rom_solver_ecsw",
        "hyper_rom_solver_ecm",
    )

    candidates = []
    for obj in module.__dict__.values():
        if not inspect.isclass(obj) or obj is Problem:
            continue

        try:
            is_problem_subclass = issubclass(obj, Problem)
        except TypeError:
            is_problem_subclass = False

        has_problem_methods = all(callable(getattr(obj, m, None)) for m in required_methods)

        if (is_problem_subclass or has_problem_methods) and not inspect.isabstract(obj):
            candidates.append(obj)

    unique_candidates = []
    for cls in candidates:
        if cls not in unique_candidates:
            unique_candidates.append(cls)

    if len(unique_candidates) == 1:
        PROBLEM_REGISTRY[name] = unique_candidates[0]


def _load_local_problem_def(problem_name: str) -> None:
    """
    Import ``problem_def.py`` from the active problem folder.

    Importing the file executes any ``@register_problem(...)`` decorator.
    The module is loaded under a unique name so notebook re-runs do not reuse a
    stale ``problem_def`` module from a previous problem folder.
    """
    _unify_masterclass_module_aliases()
    _add_sys_path(PROBLEM_DIR)
    _add_sys_path(PROBLEM_DIR.parent)
    _add_sys_path(cwd)

    problem_file = PROBLEM_DIR / "problem_def.py"
    if not problem_file.exists():
        return

    module_name = f"_skrom_parallel_problem_def_{problem_name}_{abs(hash(str(problem_file))) % (10**10)}"

    spec = importlib.util.spec_from_file_location(module_name, problem_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for '{problem_file}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Could not import local problem definition from '{problem_file}'. "
            "Check problem_def.py for import errors and make sure it uses "
            f"@register_problem('{problem_name}'), or defines exactly one concrete Problem class."
        ) from exc

    _merge_known_registries()
    _auto_register_problem_from_module(module, problem_name)


def ensure_problem_registered(problem_name: str) -> None:
    """
    Ensure that the requested problem is present in ``PROBLEM_REGISTRY``.

    This handles the common case where the problem class lives in the local
    ``problem_def.py`` file and has not been imported yet.
    """
    if problem_name in PROBLEM_REGISTRY:
        return

    attempted: List[str] = []
    errors: List[str] = []

    try:
        _load_local_problem_def(problem_name)
    except Exception as exc:
        errors.append(f"{PROBLEM_DIR / 'problem_def.py'}: {type(exc).__name__}: {exc}")

    if problem_name in PROBLEM_REGISTRY:
        return

    _unify_masterclass_module_aliases()
    _add_sys_path(PROBLEM_DIR)
    _add_sys_path(PROBLEM_DIR.parent)
    _add_sys_path(cwd)

    def _try_import(modname: str) -> None:
        attempted.append(modname)
        try:
            mod = importlib.import_module(modname)
            _merge_registry_from_module_name(modname)
            _auto_register_problem_from_module(mod, problem_name)
        except Exception as exc:
            errors.append(f"{modname}: {type(exc).__name__}: {exc}")

    # Compatibility attempts for older examples.
    for modname in (
        "problem_def",
        problem_name,
        f"{problem_name}.problem_def",
        f"skrom.problem_classes.static.{problem_name}",
    ):
        if problem_name in PROBLEM_REGISTRY:
            return
        _try_import(modname)

    if problem_name in PROBLEM_REGISTRY:
        return

    raise ValueError(
        f"Problem '{problem_name}' not registered.\n"
        f"Registry keys currently visible: {list(PROBLEM_REGISTRY.keys())}\n"
        f"Detected problem directory: {PROBLEM_DIR}\n"
        f"Attempted imports: {attempted}\n"
        "Import errors:\n  - " + ("\n  - ".join(errors) if errors else "None") + "\n"
        f"Make sure the local problem class is decorated with "
        f"@register_problem('{problem_name}'), or that problem_def.py defines exactly one concrete problem class."
    )


def get_problem(name: str) -> Problem:
    ensure_problem_registered(name)
    try:
        return PROBLEM_REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown problem '{name}'. Available problems: {list(PROBLEM_REGISTRY.keys())}")


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
    hyper_ecm_solver  = prob.hyper_rom_solver_ecm
    return (
        parameters, a, l, domain_, properties,
        fom_solver, rom_solver, hyper_deim_solver, hyper_ecsw_solver, hyper_ecm_solver
    )


# ─────────────────────────────────────────────────────────────
# PARALLEL SWEEP CORE (THREADS ONLY)
# ─────────────────────────────────────────────────────────────
class _ThreadedSweepMixin:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._cur_itr_var = contextvars.ContextVar("cur_itr", default=None)
        self._print_lock = threading.Lock()
        self.logger = logger or LOGGER

    @property
    def cur_itr(self) -> Optional[int]:
        return self._cur_itr_var.get()

    @cur_itr.setter
    def cur_itr(self, v: int) -> None:
        self._cur_itr_var.set(v)

    @contextmanager
    def _set_cur_itr(self, v: int):
        token = self._cur_itr_var.set(v)
        try:
            yield
        finally:
            self._cur_itr_var.reset(token)

    @staticmethod
    def _copy_solution(sol: Any) -> Any:
        if isinstance(sol, tuple):
            return tuple(np.copy(x) if isinstance(x, np.ndarray) else x for x in sol)
        if isinstance(sol, np.ndarray):
            return np.copy(sol)
        return sol

    def _log(self, msg: str, level: int = logging.INFO, verbose: bool = True):
        if not verbose:
            return
        with self._print_lock:
            if self.logger is not None:
                self.logger.log(level, msg)
            else:
                print(msg)


    def _threaded_sweep(
        self,
        params: np.ndarray,
        worker_fn,
        *,
        global_indices: Optional[np.ndarray] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        label: str = "Snap",
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
    ) -> Tuple[List[Any], List[float]]:
        n = len(params)
        if global_indices is None:
            global_indices = np.arange(n, dtype=int)
        else:
            global_indices = np.asarray(global_indices, dtype=int)
            if len(global_indices) != n:
                raise ValueError("global_indices must have same length as params")

        outs: List[Any] = [None] * n
        times: List[float] = [0.0] * n

        def run_one(i_local: int):
            i_global = int(global_indices[i_local])
            param = params[i_local]
            print(f"Running {param=}")

            last_exc = None
            for attempt in range(max_retries):
                try:
                    with self._set_cur_itr(i_local):
                        t0 = time.perf_counter()
                        out = worker_fn(param, i_local, i_global)
                        dt = time.perf_counter() - t0
                    return i_local, i_global, param, self._copy_solution(out), dt, None
                except BaseException as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)

            return i_local, i_global, param, None, 0.0, last_exc

        # ---- Sequential ----
        if not parallel:
            for i_local in range(n):
                iL, iG, p, out, dt, exc = run_one(i_local)
                if exc is not None:
                    self._log(f"{label} {iL+1}/{n} (global={iG}) params={p} ERROR: {exc}",
                            logging.ERROR, verbose)
                    if fail_fast:
                        raise exc
                else:
                    outs[iL] = out
                    times[iL] = dt
                    self._log(f"{label} {iL+1}/{n} (global={iG}) params={p} time={dt:.3f}s",
                            logging.INFO, verbose)
            return outs, times

        # ---- Parallel (collect results, then log in input order) ----
        results: List[Optional[Tuple[int, np.ndarray, Any, float, Optional[BaseException]]]] = [None] * n

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one, i_local) for i_local in range(n)]
            try:
                for fut in as_completed(futs):
                    iL, iG, p, out, dt, exc = fut.result()
                    if exc is not None and fail_fast:
                        for f in futs:
                            f.cancel()
                        # log the failure once
                        self._log(f"{label} {iL+1}/{n} (global={iG}) params={p} ERROR: {exc}",
                                logging.ERROR, verbose)
                        raise exc
                    results[iL] = (iG, p, out, dt, exc)
            except:
                for f in futs:
                    f.cancel()
                raise

        # emit logs in local index order + fill arrays in local index order
        for iL in range(n):
            iG, p, out, dt, exc = results[iL]  # type: ignore[misc]
            if exc is not None:
                self._log(f"{label} {iL+1}/{n} (global={iG}) params={p} ERROR: {exc}",
                        logging.ERROR, verbose)
            else:
                outs[iL] = out
                times[iL] = dt
                self._log(f"{label} {iL+1}/{n} (global={iG}) params={p} time={dt:.3f}s",
                        logging.INFO, verbose)

        return outs, times

    def _run_one_serial_with_retries(
        self,
        worker,
        param,
        i_local: int,
        i_global: int,
        *,
        max_retries: int,
        retry_delay: float,
        fail_fast: bool,
        label: str,
        verbose: bool,
    ):
        """
        Run one parameter case in the main thread before launching the threaded sweep.

        This is useful for solvers that perform one-time initialization, caching, JIT
        compilation, library setup, or file-system setup on the first call.
        """
        last_exc = None

        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()

            try:
                # Set cur_itr for the serial-first run, just like _threaded_sweep does.
                with self._set_cur_itr(i_local):
                    out = worker(param, i_local, i_global)
                dt = time.perf_counter() - t0

                if verbose:
                    print(f"[{label}] serial warm-up done (i={i_local}) in {dt:.3e}s")

                return out, dt

            except Exception as e:
                dt = time.perf_counter() - t0
                last_exc = e

                if self.logger is not None:
                    self.logger.exception(
                        f"[{label}] serial warm-up failed "
                        f"(i={i_local}, attempt={attempt + 1}/{max_retries + 1})"
                    )

                if verbose:
                    print(
                        f"[{label}] serial warm-up failed "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    traceback.print_exc()

                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue

                if fail_fast:
                    raise

                return None, dt

        if fail_fast and last_exc is not None:
            raise last_exc

        return None, 0.0


    def _threaded_sweep_serial_first(
        self,
        params,
        worker,
        *,
        global_indices,
        parallel: bool,
        serial_first: bool,
        max_workers,
        verbose: bool,
        label: str,
        max_retries: int,
        retry_delay: float,
        fail_fast: bool,
    ):
        """
        Run the first parameter case serially and run the remaining cases in threads.

        If ``serial_first`` is False, or ``parallel`` is False, this falls back to
        ``_threaded_sweep`` and preserves the previous behavior.
        """
        n = len(params)

        if n == 0:
            return [], []

        if (not parallel) or (not serial_first) or (n == 1):
            return self._threaded_sweep(
                params,
                worker,
                global_indices=global_indices,
                parallel=parallel,
                max_workers=max_workers,
                verbose=verbose,
                label=label,
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

        outs = [None] * n
        times = [0.0] * n

        # 1) Run the first case serially in the main thread.
        outs[0], times[0] = self._run_one_serial_with_retries(
            worker,
            params[0],
            0,
            global_indices[0],
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
            label=label,
            verbose=verbose,
        )

        # 2) Run the remaining cases in parallel. Shift the local index by one so
        # downstream code still sees the same local indexing as the full parameter list.
        def worker_rest(param, i_local_rest, i_global_rest):
            return worker(param, i_local_rest + 1, i_global_rest)

        outs_rest, times_rest = self._threaded_sweep(
            params[1:],
            worker_rest,
            global_indices=global_indices[1:],
            parallel=True,
            max_workers=max_workers,
            verbose=verbose,
            label=label,
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        outs[1:] = outs_rest
        times[1:] = times_rest

        return outs, times

# ─────────────────────────────────────────────────────────────
# OFFLINE SNAPSHOT GENERATION
# ─────────────────────────────────────────────────────────────
import os
import re
import json
import time
import hashlib
import tempfile
from pathlib import Path

import numpy as np


class fom_simulation(_ThreadedSweepMixin):
    def __init__(self, num_snapshots: int = 32, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        prob = get_problem(PROBLEM)
        (
            self.parameters,
            self.bilinear_forms,
            self.linear_forms,
            self.domain,
            self.properties,
            self.fom_solver,
            *_,
        ) = assign_properties(prob)

        self._baseline_attrs = set(vars(self))

        data = self.domain()
        self.mesh = data["mesh"]
        self.basis = data["basis"]

        self.num_snapshots = num_snapshots
        (
            self.param_list,
            self.param_range,
            self.train_mask,
            self.test_mask,
        ) = self.parameters(num_snapshots)

        self.fos_solutions: Any = []
        self.fos_time: List[float] = []


    def run_simulation(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
        serial_first: bool = False,
    ) -> None:
        cur_dir = Path(os.getcwd())
        ckpt_dir = cur_dir / "fos_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        def _problem_tag() -> str:
            return (
                getattr(self, "PROBLEM_NAME", None)
                or getattr(self, "problem_name", None)
                or self.__class__.__name__
            )

        def _param_bytes(param) -> bytes:
            # numeric params -> stable bytes; fallback -> stable JSON bytes
            try:
                arr = np.asarray(param, dtype=float).ravel()
                return arr.tobytes()
            except Exception:
                js = json.dumps(param, sort_keys=True, default=str)
                return js.encode("utf-8")

        def _param_store(param):
            # store param in file for inspection
            try:
                return np.asarray(param, dtype=float).ravel()
            except Exception:
                js = json.dumps(param, sort_keys=True, default=str)
                return np.array([js], dtype="U")

        def _param_hash(param) -> str:
            return hashlib.sha1(_param_bytes(param)).hexdigest()[:16]

        def _ckpt_path(param) -> Path:
            h = _param_hash(param)
            # short filename to avoid Windows path issues
            return ckpt_dir / f"fos_{_problem_tag()}_h{h}.npz"

        def _atomic_save_npz(path: Path, payload: dict):
            # Use a short temp prefix and Windows extended-length paths to avoid
            # FileNotFoundError/OSError in long OneDrive project folders.
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            fd = None
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=_win_long_path(path.parent),
                    prefix="tmp_",
                    suffix=".npz",
                )
                os.close(fd)
                fd = None

                np.savez_compressed(tmp_path, **payload)
                os.replace(tmp_path, _win_long_path(path))

            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                if tmp_path is not None and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        def _save_solution(path: Path, sol, param, elapsed: float, snap_index: int):
            payload = {
                "snap_index": np.array([snap_index], dtype=np.int64),
                "solve_time": np.array([elapsed], dtype=float),
                "param": _param_store(param),
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

        def worker(param, i_local, i_global):
            path = _ckpt_path(param)

            # resume
            if path.exists():
                sol, t_saved = _load_solution(path)
                return sol, t_saved

            # compute + checkpoint
            t0 = time.perf_counter()
            sol = self.fom_solver(cls=self, param=param)
            elapsed = time.perf_counter() - t0

            _save_solution(path, sol, param, elapsed, snap_index=int(i_global))
            return sol, elapsed

        results, _times_measured = self._threaded_sweep_serial_first(
            self.param_list,
            worker,
            global_indices=np.arange(len(self.param_list)),
            parallel=parallel,
            serial_first=serial_first,
            max_workers=max_workers,
            verbose=verbose,
            label="FOM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        if any(r is None for r in results):
            bad = [i for i, r in enumerate(results) if r is None]
            raise RuntimeError(f"FOM had failed snapshots at indices: {bad}.")

        sols = [r[0] for r in results]
        self.fos_time = [float(r[1]) for r in results]

        if len(sols) == 0:
            return

        first = sols[0]
        if isinstance(first, tuple):
            self.fos_solutions = np.array(sols, dtype=object)

            train_ids = np.where(self.train_mask)[0]
            ncomp = len(first)
            train_ref = []
            for j in range(ncomp):
                stack_j = np.stack([self.fos_solutions[k][j] for k in train_ids], axis=0)
                train_ref.append(np.mean(stack_j, axis=0))
            self.train_ref = tuple(train_ref)
        else:
            self.fos_solutions = np.asarray(sols)
            self.train_ref = np.mean(self.fos_solutions[self.train_mask], axis=0)

        new = set(vars(self)) - self._baseline_attrs
        save_dict = {k: getattr(self, k) for k in new if not k.startswith("_")}
        rom_data_gen(save_dict, str(cur_dir))

# ─────────────────────────────────────────────────────────────
# ONLINE ROM EVALUATION
# ─────────────────────────────────────────────────────────────
class rom_simulation(_ThreadedSweepMixin):
    def __init__(
        self,
        train_ref=None, test_ref = None, fos_solutions=None,
        train_mask=None, test_mask=None,
        V_sel=None, n_sel=None, N_rom_snap=None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger=logger)

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
            self.hyper_rom_solver_ecm,
        ) = assign_properties(prob)

        cur_dir = os.getcwd()
        rom_dir = os.path.join(cur_dir, "ROM_data")
        load_rom_data(self, rom_dir)

        if train_ref is not None:
            self.train_ref = train_ref
        if test_ref is not None:
            self.test_ref = test_ref
        if fos_solutions is not None:
            self.fos_solutions = fos_solutions
        if train_mask is not None:
            self.train_mask = train_mask
        if test_mask is not None:
            self.test_mask = test_mask

        self.param_list_test = self.param_list[self.test_mask]
        self.fos_test_data   = self.fos_solutions[self.test_mask].astype(np.float32, copy=False)
        self.fos_test_time   = np.asarray(self.fos_time)[self.test_mask]


        self.sol_train_ms    = self.fos_solutions[self.train_mask].astype(np.float32, copy=False) - self.train_ref.astype(np.float32, copy=False)

        self.V_sel      = V_sel
        self.n_sel      = n_sel
        self.N_rom_snap = N_rom_snap or len(self.param_list_test)

        unwrap_attr(self, "basis")
        unwrap_attr(self, "mesh")

        self._test_global_idx = np.flatnonzero(self.test_mask)

    def _ref_for(self, i_local_test: int, i_global_param: int):
        if hasattr(self.test_ref, "ndim") and self.test_ref.ndim == 3:
            if self.test_ref.shape[0] == len(self.param_list):
                return self.test_ref[i_global_param]
            if self.test_ref.shape[0] == len(self.param_list_test):
                return self.test_ref[i_local_test]
            return self.test_ref[i_local_test]
        return self.test_ref

    @staticmethod
    def _maybe_transpose_like(sol_candidate: np.ndarray, sol_ref: np.ndarray) -> np.ndarray:
        if sol_candidate.shape != sol_ref.shape and sol_candidate.ndim == 2 and sol_candidate.T.shape == sol_ref.shape:
            return sol_candidate.T
        return sol_candidate


    def _run_one_serial_with_retries(
        self,
        worker,
        param,
        i_local: int,
        i_global: int,
        *,
        max_retries: int,
        retry_delay: float,
        fail_fast: bool,
        label: str,
        verbose: bool,
    ):
        last_exc = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                # Set cur_itr for the serial-first run, just like _threaded_sweep does.
                with self._set_cur_itr(i_local):
                    out = worker(param, i_local, i_global)
                dt = time.perf_counter() - t0
                if verbose:
                    print(f"[{label}] serial warm-up done (i={i_local}) in {dt:.3e}s")
                return out, dt
            except Exception as e:
                dt = time.perf_counter() - t0
                last_exc = e
                if self.logger is not None:
                    self.logger.exception(
                        f"[{label}] serial warm-up failed (i={i_local}, attempt={attempt+1}/{max_retries+1})"
                    )
                if verbose:
                    print(f"[{label}] serial warm-up failed (attempt {attempt+1}/{max_retries+1})")
                    traceback.print_exc()

                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue

                if fail_fast:
                    raise
                return None, dt  # keep alignment; caller already handles outs[i] is None

        if fail_fast and last_exc is not None:
            raise last_exc
        return None, 0.0

    def _threaded_sweep_serial_first(
        self,
        params,
        worker,
        *,
        global_indices,
        parallel: bool,
        serial_first: bool,
        max_workers,
        verbose: bool,
        label: str,
        max_retries: int,
        retry_delay: float,
        fail_fast: bool,
    ):
        n = len(params)
        if n == 0:
            return [], []

        # default: your existing behavior
        if (not parallel) or (not serial_first) or (n == 1):
            return self._threaded_sweep(
                params,
                worker,
                global_indices=global_indices,
                parallel=parallel,
                max_workers=max_workers,
                verbose=verbose,
                label=label,
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

        # allocate full outputs
        outs = [None] * n
        times = [0.0] * n

        # 1) run first item serially in the main thread
        outs[0], times[0] = self._run_one_serial_with_retries(
            worker,
            params[0],
            0,
            global_indices[0],
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
            label=label,
            verbose=verbose,
        )

        # 2) run the remaining items with your threaded sweep
        def worker_rest(param, i_local_rest, i_global_rest):
            # IMPORTANT: shift local index by +1 to keep alignment with fos_test_data
            return worker(param, i_local_rest + 1, i_global_rest)

        outs_rest, times_rest = self._threaded_sweep(
            params[1:],
            worker_rest,
            global_indices=global_indices[1:],
            parallel=True,  # force parallel for the remainder
            max_workers=max_workers,
            verbose=verbose,
            label=label,
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        outs[1:] = outs_rest
        times[1:] = times_rest
        return outs, times


    def run_rom_simulation(
        self,
        full_order: bool = True,
        new_params = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
        serial_first: bool = False,
    ):
        self.speed_up      = []
        self.rom_error     = []
        self.rom_solutions = []

        if new_params is None:
            if self.test_ref is None:
                raise ValueError("self.test_ref must not be None when evaluating ROM error on the test set.")

            n = min(self.N_rom_snap, len(self.param_list_test))
            params = self.param_list_test[:n]
            global_idx = self._test_global_idx[:n]

            def worker(param, i_local, i_global):
                self.cur_itr = i_local
                sol_red_ = self.rom_solver(cls=self, param=param)
                sol_rom = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

                sol_fos = self.fos_test_data[i_local]
                sol_rom = self._maybe_transpose_like(sol_rom, sol_fos)
                if sol_rom.shape != sol_fos.shape:
                    raise ValueError(f"shape mismatch: fos{sol_fos.shape}, rom{sol_rom.shape}")

                err = 100.0 * np.linalg.norm(sol_fos - sol_rom) / np.linalg.norm(sol_fos)
                sol_store = self._copy_solution(sol_rom if full_order else sol_red_)
                return sol_store, float(err)

            outs, times = self._threaded_sweep_serial_first(
                params,
                worker,
                global_indices=global_idx,
                parallel=parallel,
                serial_first=serial_first,
                max_workers=max_workers,
                verbose=verbose,
                label="ROM",
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

            rom_solutions: List[Any] = [None] * n
            rom_error: List[float] = [np.nan] * n
            speed_up: List[float] = [np.nan] * n

            for i in range(n):
                if outs[i] is None:
                    continue
                sol_store, err = outs[i]
                rom_solutions[i] = sol_store
                rom_error[i] = err
                speed_up[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

            self.rom_solutions = rom_solutions
            self.rom_error = rom_error
            self.speed_up = speed_up
            return self.rom_error, self.speed_up

        params = list(new_params)
        n = len(params)
        global_idx = np.arange(n, dtype=int)

        if full_order and self.test_ref is None:
            raise ValueError("self.test_ref must not be None when full_order=True and new_params is provided.")

        def worker(param, i_local, i_global):
            self.cur_itr = i_local
            sol_red_ = self.rom_solver(cls=self, param=param)

            if full_order:
                if hasattr(self.test_ref, "ndim") and self.test_ref.ndim == 3:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i_local])
                else:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)
            else:
                sol_out = sol_red_

            return self._copy_solution(sol_out)

        outs, _ = self._threaded_sweep_serial_first(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            serial_first=serial_first,
            max_workers=max_workers,
            verbose=verbose,
            label="ROM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        rom_solutions: List[Any] = [None] * n
        for i in range(n):
            if outs[i] is None:
                continue
            rom_solutions[i] = outs[i]

        self.rom_solutions = rom_solutions
        return self.rom_solutions

    def run_hyper_rom_simulation_ecsw(
        self,
        z,
        full_order: bool = True,
        new_params = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
        serial_first: bool = False,
    ):
        self.hyper_speed_up      = []
        self.hyper_rom_error     = []
        self.hyper_rom_solutions = []
        self.z = z

        if new_params is None:
            if self.test_ref is None:
                raise ValueError("self.test_ref must not be None when evaluating ECSW hyper-ROM error on the test set.")

            n = min(self.N_rom_snap, len(self.param_list_test))
            params = self.param_list_test[:n]
            global_idx = self._test_global_idx[:n]

            def worker(param, i_local, i_global):
                self.cur_itr = i_local
                sol_red_ = self.hyper_rom_solver_ecsw(cls=self, param=param)
                sol_hyp = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

                sol_fos = self.fos_test_data[i_local]
                sol_hyp = self._maybe_transpose_like(sol_hyp, sol_fos)
                if sol_hyp.shape != sol_fos.shape:
                    raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyp.shape}")

                err = 100.0 * np.linalg.norm(sol_fos - sol_hyp) / np.linalg.norm(sol_fos)
                sol_store = self._copy_solution(sol_hyp if full_order else sol_red_)
                return sol_store, float(err)

            outs, times = self._threaded_sweep_serial_first(
                params,
                worker,
                global_indices=global_idx,
                parallel=parallel,
                serial_first=serial_first,
                max_workers=max_workers,
                verbose=verbose,
                label="ECSW",
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

            hyp_solutions: List[Any] = [None] * n
            hyp_error: List[float] = [np.nan] * n
            hyp_speed: List[float] = [np.nan] * n

            for i in range(n):
                if outs[i] is None:
                    continue
                sol_store, err = outs[i]
                hyp_solutions[i] = sol_store
                hyp_error[i] = err
                hyp_speed[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

            self.hyper_rom_solutions = hyp_solutions
            self.hyper_rom_error = hyp_error
            self.hyper_speed_up = hyp_speed
            return self.hyper_rom_error, self.hyper_speed_up

        params = list(new_params)
        n = len(params)
        global_idx = np.arange(n, dtype=int)

        if full_order and self.test_ref is None:
            raise ValueError("self.test_ref must not be None when full_order=True and new_params is provided.")

        def worker(param, i_local, i_global):
            self.cur_itr = i_local
            sol_red_ = self.hyper_rom_solver_ecsw(cls=self, param=param)

            if full_order:
                if hasattr(self.test_ref, "ndim") and self.test_ref.ndim == 3:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i_local])
                else:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)
            else:
                sol_out = sol_red_

            return self._copy_solution(sol_out)

        outs, _ = self._threaded_sweep_serial_first(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            serial_first=serial_first,
            max_workers=max_workers,
            verbose=verbose,
            label="ECSW",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        hyp_solutions: List[Any] = [None] * n
        for i in range(n):
            if outs[i] is None:
                continue
            hyp_solutions[i] = outs[i]

        self.hyper_rom_solutions = hyp_solutions
        return self.hyper_rom_solutions

    def run_hyper_rom_simulation_ecm(
        self,
        z,
        full_order: bool = True,
        new_params = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
        serial_first: bool = False,
    ):
        self.hyper_speed_up      = []
        self.hyper_rom_error     = []
        self.hyper_rom_solutions = []
        self.z = z

        if new_params is None:
            if self.test_ref is None:
                raise ValueError("self.test_ref must not be None when evaluating ECM hyper-ROM error on the test set.")

            n = min(self.N_rom_snap, len(self.param_list_test))
            params = self.param_list_test[:n]
            global_idx = self._test_global_idx[:n]

            def worker(param, i_local, i_global):
                self.cur_itr = i_local
                sol_red_ = self.hyper_rom_solver_ecm(cls=self, param=param)
                sol_hyp = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

                sol_fos = self.fos_test_data[i_local]
                sol_hyp = self._maybe_transpose_like(sol_hyp, sol_fos)
                if sol_hyp.shape != sol_fos.shape:
                    raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyp.shape}")

                err = 100.0 * np.linalg.norm(sol_fos - sol_hyp) / np.linalg.norm(sol_fos)
                sol_store = self._copy_solution(sol_hyp if full_order else sol_red_)
                return sol_store, float(err)

            outs, times = self._threaded_sweep_serial_first(
                params,
                worker,
                global_indices=global_idx,
                parallel=parallel,
                serial_first=serial_first,
                max_workers=max_workers,
                verbose=verbose,
                label="ECM",
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

            hyp_solutions: List[Any] = [None] * n
            hyp_error: List[float] = [np.nan] * n
            hyp_speed: List[float] = [np.nan] * n

            for i in range(n):
                if outs[i] is None:
                    continue
                sol_store, err = outs[i]
                hyp_solutions[i] = sol_store
                hyp_error[i] = err
                hyp_speed[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

            self.hyper_rom_solutions = hyp_solutions
            self.hyper_rom_error = hyp_error
            self.hyper_speed_up = hyp_speed
            return self.hyper_rom_error, self.hyper_speed_up

        params = list(new_params)
        n = len(params)
        global_idx = np.arange(n, dtype=int)

        if full_order and self.test_ref is None:
            raise ValueError("self.test_ref must not be None when full_order=True and new_params is provided.")

        def worker(param, i_local, i_global):
            self.cur_itr = i_local
            sol_red_ = self.hyper_rom_solver_ecm(cls=self, param=param)

            if full_order:
                if hasattr(self.test_ref, "ndim") and self.test_ref.ndim == 3:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i_local])
                else:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)
            else:
                sol_out = sol_red_

            return self._copy_solution(sol_out)

        outs, _ = self._threaded_sweep_serial_first(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            serial_first=serial_first,
            max_workers=max_workers,
            verbose=verbose,
            label="ECM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        hyp_solutions: List[Any] = [None] * n
        for i in range(n):
            if outs[i] is None:
                continue
            hyp_solutions[i] = outs[i]

        self.hyper_rom_solutions = hyp_solutions
        return self.hyper_rom_solutions

    def run_hyper_rom_simulation_deim(
        self,
        z,
        deim_mat,
        sampled_rows,
        full_order: bool = True,
        new_params = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
        serial_first: bool = False,
    ):
        self.hyper_speed_up      = []
        self.hyper_rom_error     = []
        self.hyper_rom_solutions = []
        self.z = z
        self.deim_mat = deim_mat
        self.sampled_rows = sampled_rows

        if new_params is None:
            if self.test_ref is None:
                raise ValueError("self.test_ref must not be None when evaluating DEIM hyper-ROM error on the test set.")

            n = min(self.N_rom_snap, len(self.param_list_test))
            params = self.param_list_test[:n]
            global_idx = self._test_global_idx[:n]

            def worker(param, i_local, i_global):
                self.cur_itr = i_local
                sol_red_ = self.hyper_rom_solver_deim(cls=self, param=param)
                sol_hyp = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

                sol_fos = self.fos_test_data[i_local]
                sol_hyp = self._maybe_transpose_like(sol_hyp, sol_fos)
                if sol_hyp.shape != sol_fos.shape:
                    raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyp.shape}")

                err = 100.0 * np.linalg.norm(sol_fos - sol_hyp) / np.linalg.norm(sol_fos)
                sol_store = self._copy_solution(sol_hyp if full_order else sol_red_)
                return sol_store, float(err)

            outs, times = self._threaded_sweep_serial_first(
                params,
                worker,
                global_indices=global_idx,
                parallel=parallel,
                serial_first=serial_first,
                max_workers=max_workers,
                verbose=verbose,
                label="DEIM",
                max_retries=max_retries,
                retry_delay=retry_delay,
                fail_fast=fail_fast,
            )

            hyp_solutions: List[Any] = [None] * n
            hyp_error: List[float] = [np.nan] * n
            hyp_speed: List[float] = [np.nan] * n

            for i in range(n):
                if outs[i] is None:
                    continue
                sol_store, err = outs[i]
                hyp_solutions[i] = sol_store
                hyp_error[i] = err
                hyp_speed[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

            self.hyper_rom_solutions = hyp_solutions
            self.hyper_rom_error = hyp_error
            self.hyper_speed_up = hyp_speed
            return self.hyper_rom_error, self.hyper_speed_up

        params = list(new_params)
        n = len(params)
        global_idx = np.arange(n, dtype=int)

        if full_order and self.test_ref is None:
            raise ValueError("self.test_ref must not be None when full_order=True and new_params is provided.")

        def worker(param, i_local, i_global):
            self.cur_itr = i_local
            sol_red_ = self.hyper_rom_solver_deim(cls=self, param=param)

            if full_order:
                if hasattr(self.test_ref, "ndim") and self.test_ref.ndim == 3:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref[i_local])
                else:
                    sol_out = reconstruct_solution(sol_red_, self.V_sel, self.test_ref)
            else:
                sol_out = sol_red_

            return self._copy_solution(sol_out)

        outs, _ = self._threaded_sweep_serial_first(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            serial_first=serial_first,
            max_workers=max_workers,
            verbose=verbose,
            label="DEIM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        hyp_solutions: List[Any] = [None] * n
        for i in range(n):
            if outs[i] is None:
                continue
            hyp_solutions[i] = outs[i]

        self.hyper_rom_solutions = hyp_solutions
        return self.hyper_rom_solutions
