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

import numpy as np

from skrom.fom.fem_utils import unwrap_attr
from skrom.rom.rom_utils import rom_data_gen, load_rom_data, reconstruct_solution


# ─────────────────────────────────────────────────────────────
# BASIC LOGGER (optional)
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# PROBLEM DETECTION (keep your workflow; allow env override too)
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


# IMPORTANT: reuse shared registry if it exists (your problems likely register there)
_USING_SHARED_REGISTRY = False
try:
    import skrom.problem_classes.static.master_class as _mc
    PROBLEM_REGISTRY = _mc.PROBLEM_REGISTRY  # shared dict reference
    register_problem = _mc.register_problem  # shared decorator
    _USING_SHARED_REGISTRY = True
except Exception:
    PROBLEM_REGISTRY: Dict[str, Type[Problem]] = {}
    def register_problem(name: str):
        def deco(cls: Type[Problem]) -> Type[Problem]:
            PROBLEM_REGISTRY[name] = cls
            return cls
        return deco


def _add_sys_path(p: Path) -> None:
    sp = str(p)
    if sp and sp not in sys.path:
        sys.path.insert(0, sp)


def _merge_registry_from_module(modname: str) -> None:
    """Merge any PROBLEM_REGISTRY found in an imported module into ours."""
    mod = sys.modules.get(modname)
    if mod is None:
        return
    reg = getattr(mod, "PROBLEM_REGISTRY", None)
    if isinstance(reg, dict) and reg:
        PROBLEM_REGISTRY.update(reg)  # if shared, updates master_class registry too


def ensure_problem_registered(problem_name: str) -> None:
    """
    Import the module that defines the problem (so @register_problem runs).

    Supports:
      - local file: ./problem_name.py  (when cwd is repo root OR problem folder)
      - local package: ./problem_name/__init__.py
      - package module: skrom.problem_classes.static.problem_name
    """
    if problem_name in PROBLEM_REGISTRY:
        return

    cwd = Path.cwd()

    # Make cwd and parent importable (covers running inside problem folder)
    _add_sys_path(cwd)
    _add_sys_path(cwd.parent)

    attempted: List[str] = []
    errors: List[str] = []

    def _try(modname: str) -> None:
        attempted.append(modname)
        try:
            importlib.import_module(modname)
            _merge_registry_from_module(modname)
        except Exception as e:
            errors.append(f"{modname}: {type(e).__name__}: {e}")

    # 1) Try direct import by name (works if problem_name.py exists on sys.path,
    #    or if problem_name is a package folder with __init__.py)
    _try(problem_name)
    if problem_name in PROBLEM_REGISTRY:
        return

    # 2) If cwd contains folder ./problem_name, try importing it as package
    if (cwd / problem_name).is_dir():
        _add_sys_path(cwd)
        _try(problem_name)
        if problem_name in PROBLEM_REGISTRY:
            return

    # 3) Try package namespace
    _try(f"skrom.problem_classes.static.{problem_name}")
    if problem_name in PROBLEM_REGISTRY:
        return

    raise ValueError(
        f"Problem '{problem_name}' not registered.\n"
        f"Registry keys currently visible: {list(PROBLEM_REGISTRY.keys())}\n"
        f"Using shared registry from master_class: {_USING_SHARED_REGISTRY}\n"
        f"Attempted imports: {attempted}\n"
        "Import errors:\n  - " + "\n  - ".join(errors)
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
    return (
        parameters, a, l, domain_, properties,
        fom_solver, rom_solver, hyper_deim_solver, hyper_ecsw_solver
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
            path.parent.mkdir(parents=True, exist_ok=True)

            # IMPORTANT: suffix ".npz" so numpy does not append ".npz" again
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix="tmp_fos_",
                suffix=".npz",
            )
            os.close(fd)

            try:
                np.savez_compressed(tmp_path, **payload)
                os.replace(tmp_path, str(path))
            finally:
                if os.path.exists(tmp_path):
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

        results, _times_measured = self._threaded_sweep(
            self.param_list,
            worker,
            global_indices=np.arange(len(self.param_list)),
            parallel=parallel,
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

    def run_rom_simulation(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
    ):
        self.speed_up      = []
        self.rom_error     = []
        self.rom_solutions = []

        n = min(self.N_rom_snap, len(self.param_list_test))
        params = self.param_list_test[:n]
        global_idx = self._test_global_idx[:n]

        def worker(param, i_local, i_global):
            sol_red_ = self.rom_solver(cls=self, param=param)
            sol_rom = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

            sol_fos = self.fos_test_data[i_local]  # FIX
            sol_rom = self._maybe_transpose_like(sol_rom, sol_fos)
            if sol_rom.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, rom{sol_rom.shape}")

            err = 100.0 * np.linalg.norm(sol_fos - sol_rom) / np.linalg.norm(sol_fos)
            return sol_rom.copy(), float(err)

        outs, times = self._threaded_sweep(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            max_workers=max_workers,
            verbose=verbose,
            label="ROM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        rom_solutions: List[np.ndarray] = [None] * n  # type: ignore
        rom_error: List[float] = [np.nan] * n
        speed_up: List[float] = [np.nan] * n

        for i in range(n):
            if outs[i] is None:
                continue
            sol_rom, err = outs[i]
            rom_solutions[i] = sol_rom
            rom_error[i] = err
            speed_up[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

        self.rom_solutions = rom_solutions
        self.rom_error = rom_error
        self.speed_up = speed_up
        return self.rom_error, self.speed_up

    def run_hyper_rom_simulation_ecsw(
        self,
        z,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
    ):
        self.hyper_speed_up      = []
        self.hyper_rom_error     = []
        self.hyper_rom_solutions = []
        self.z = z

        n = min(self.N_rom_snap, len(self.param_list_test))
        params = self.param_list_test[:n]
        global_idx = self._test_global_idx[:n]

        def worker(param, i_local, i_global):
            sol_red_ = self.hyper_rom_solver_ecsw(cls=self, param=param)
            sol_hyp = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

            sol_fos = self.fos_test_data[i_local]
            sol_hyp = self._maybe_transpose_like(sol_hyp, sol_fos)
            if sol_hyp.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyp.shape}")

            err = 100.0 * np.linalg.norm(sol_fos - sol_hyp) / np.linalg.norm(sol_fos)
            return sol_hyp.copy(), float(err)

        outs, times = self._threaded_sweep(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            max_workers=max_workers,
            verbose=verbose,
            label="ECSW",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        hyp_solutions: List[np.ndarray] = [None] * n  # type: ignore
        hyp_error: List[float] = [np.nan] * n
        hyp_speed: List[float] = [np.nan] * n

        for i in range(n):
            if outs[i] is None:
                continue
            sol_hyp, err = outs[i]
            hyp_solutions[i] = sol_hyp
            hyp_error[i] = err
            hyp_speed[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

        self.hyper_rom_solutions = hyp_solutions
        self.hyper_rom_error = hyp_error
        self.hyper_speed_up = hyp_speed
        return self.hyper_rom_error, self.hyper_speed_up

    def run_hyper_rom_simulation_deim(
        self,
        z,
        deim_mat,
        sampled_rows,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        fail_fast: bool = True,
    ):
        self.hyper_speed_up      = []
        self.hyper_rom_error     = []
        self.hyper_rom_solutions = []
        self.z = z
        self.deim_mat = deim_mat
        self.sampled_rows = sampled_rows

        n = min(self.N_rom_snap, len(self.param_list_test))
        params = self.param_list_test[:n]
        global_idx = self._test_global_idx[:n]

        def worker(param, i_local, i_global):
            sol_red_ = self.hyper_rom_solver_deim(cls=self, param=param)

            # FIX: reconstruct once
            sol_hyp = reconstruct_solution(sol_red_, self.V_sel, self._ref_for(i_local, i_global))

            sol_fos = self.fos_test_data[i_local]
            sol_hyp = self._maybe_transpose_like(sol_hyp, sol_fos)
            if sol_hyp.shape != sol_fos.shape:
                raise ValueError(f"shape mismatch: fos{sol_fos.shape}, hyper{sol_hyp.shape}")

            err = 100.0 * np.linalg.norm(sol_fos - sol_hyp) / np.linalg.norm(sol_fos)
            return sol_hyp.copy(), float(err)

        outs, times = self._threaded_sweep(
            params,
            worker,
            global_indices=global_idx,
            parallel=parallel,
            max_workers=max_workers,
            verbose=verbose,
            label="DEIM",
            max_retries=max_retries,
            retry_delay=retry_delay,
            fail_fast=fail_fast,
        )

        hyp_solutions: List[np.ndarray] = [None] * n  # type: ignore
        hyp_error: List[float] = [np.nan] * n
        hyp_speed: List[float] = [np.nan] * n

        for i in range(n):
            if outs[i] is None:
                continue
            sol_hyp, err = outs[i]
            hyp_solutions[i] = sol_hyp
            hyp_error[i] = err
            hyp_speed[i] = float(self.fos_test_time[i] / times[i]) if times[i] > 0 else np.nan

        self.hyper_rom_solutions = hyp_solutions
        self.hyper_rom_error = hyp_error
        self.hyper_speed_up = hyp_speed
        return self.hyper_rom_error, self.hyper_speed_up
