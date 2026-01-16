"""
Compute and visualize ROM error metrics for flat (vectorized) data reconstructions.

This module provides:
  - compute_rom_error_metrics_flat: calculates time-dependent and global error measures
    (L2, L∞, RMSE, MAE, R², explained variance, quantiles, and optional energy norm).
  - plot_rom_error_diagnostics_flat: diagnostic plots including true vs. ROM scatter,
    spatial snapshots for selected snapshots, and raincloud plots of error and speed-up.
  - generate_rom_error_report: prints a structured summary of global and time-dependent
    ROM error statistics to the console.

Notes on shapes
---------------
Supported shapes for u and u_rom:
  - (n_snapshots, n_space)
  - (n_snapshots, n_time, n_space)

If your data uses a different axis order, pass snapshot_axis/time_axis/space_axis.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def _move_axes_to_sts(a: np.ndarray,
                      snapshot_axis: int,
                      time_axis: int | None,
                      space_axis: int) -> np.ndarray:
    """
    Return a view of `a` with axes ordered as (snapshot, time?, space).

    - If a.ndim == 2, output is (n_snap, n_space)
    - If a.ndim == 3, output is (n_snap, n_time, n_space)
    """
    a = np.asarray(a)

    if a.ndim == 2:
        # order as (snap, space)
        if time_axis is not None:
            raise ValueError("time_axis must be None for 2D arrays.")
        return np.moveaxis(a, (snapshot_axis, space_axis), (0, 1))

    if a.ndim == 3:
        if time_axis is None:
            raise ValueError("time_axis must be provided for 3D arrays.")
        return np.moveaxis(a, (snapshot_axis, time_axis, space_axis), (0, 1, 2))

    raise ValueError(f"Expected 2D or 3D array, got ndim={a.ndim}.")


def compute_rom_error_metrics_flat(u,
                                  u_rom,
                                  K: np.ndarray | None = None,
                                  *,
                                  snapshot_axis: int = 0,
                                  time_axis: int | None = None,
                                  space_axis: int = -1,
                                  eps: float = 1e-30) -> dict:
    """
    Compute various error metrics between full-order and ROM reconstructions.

    Parameters
    ----------
    u, u_rom
        Full-order and ROM fields. Supported shapes:
          - (n_snapshots, n_space)
          - (n_snapshots, n_time, n_space)
    K : array_like, optional
        Stiffness matrix for energy-norm metrics. Must be (n_space, n_space).
        If provided and u is 3D, the energy terms are accumulated over (snapshot, time).
    snapshot_axis, time_axis, space_axis
        Axis indices used if u has a different ordering.
        For 2D arrays, keep time_axis=None.
    eps
        Small number to avoid division-by-zero.

    Returns
    -------
    metrics : dict
        Contains time-dependent and global metrics. For 3D inputs, time-dependent
        arrays have shape (n_snapshots, n_time).
    """
    u = np.asarray(u)
    u_r = np.asarray(u_rom)
    if u.shape != u_r.shape:
        raise ValueError(f"u and u_rom must have the same shape. Got {u.shape} vs {u_r.shape}")

    # canonicalize to (snap, time?, space)
    if u.ndim == 2:
        U = _move_axes_to_sts(u, snapshot_axis=snapshot_axis, time_axis=None, space_axis=space_axis)
        Ur = _move_axes_to_sts(u_r, snapshot_axis=snapshot_axis, time_axis=None, space_axis=space_axis)
        has_time = False
    else:
        if time_axis is None:
            # common default for 3D: (snap, time, space)
            time_axis = 1
        U = _move_axes_to_sts(u, snapshot_axis=snapshot_axis, time_axis=time_axis, space_axis=space_axis)
        Ur = _move_axes_to_sts(u_r, snapshot_axis=snapshot_axis, time_axis=time_axis, space_axis=space_axis)
        has_time = True

    err = U - Ur
    metrics: dict = {}

    # ---------- time-dependent (per snapshot, and per time if present) ----------
    # L2 per sample in the "space" direction
    if not has_time:
        # (n_snap,)
        L2_t = np.linalg.norm(err, axis=-1)
        U_norm_t = np.linalg.norm(U, axis=-1)
        relL2_t = L2_t / np.maximum(U_norm_t, eps)
        RMSE_t = np.sqrt(np.mean(err**2, axis=-1))
        MAE_t = np.mean(np.abs(err), axis=-1)

        metrics.update({
            "L2_error_time": L2_t,
            "relative_L2_error_time": relL2_t,
            "RMSE_time": RMSE_t,
            "MAE_time": MAE_t,
            "time_avg_rel_L2_error": float(np.mean(relL2_t)),
        })
    else:
        # (n_snap, n_time)
        L2_st = np.linalg.norm(err, axis=-1)
        U_norm_st = np.linalg.norm(U, axis=-1)
        relL2_st = L2_st / np.maximum(U_norm_st, eps)
        RMSE_st = np.sqrt(np.mean(err**2, axis=-1))
        MAE_st = np.mean(np.abs(err), axis=-1)

        # convenient aggregations per snapshot (mean over time)
        relL2_snap = np.mean(relL2_st, axis=1)

        metrics.update({
            "L2_error_time": L2_st,
            "relative_L2_error_time": relL2_st,
            "RMSE_time": RMSE_st,
            "MAE_time": MAE_st,
            "relative_L2_error_snapshot_mean": relL2_snap,
            "time_avg_rel_L2_error": float(np.mean(relL2_st)),
        })

    # ---------- global metrics (flatten over snapshot/time/space) ----------
    flat_u = U.reshape(-1)
    flat_err = err.reshape(-1)
    flat_abs_err = np.abs(flat_err)

    L2 = float(np.linalg.norm(flat_err))
    relL2 = float(L2 / np.maximum(np.linalg.norm(flat_u), eps))
    Linf = float(np.max(flat_abs_err))
    relLinf = float(Linf / np.maximum(np.max(np.abs(flat_u)), eps))
    RMSE = float(np.sqrt(np.mean(flat_err**2)))
    MAE = float(np.mean(flat_abs_err))

    # R²
    ss_res = float(np.sum(flat_err**2))
    ss_tot = float(np.sum((flat_u - flat_u.mean())**2))
    R2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else float("nan")

    # explained variance: 1 - Var(err)/Var(u)
    var_u = float(np.var(flat_u))
    var_e = float(np.var(flat_err))
    expvar = float(1.0 - var_e / var_u) if var_u > eps else float("nan")

    metrics.update({
        "L2_error": L2,
        "relative_L2_error": relL2,
        "Linf_error": Linf,
        "relative_Linf_error": relLinf,
        "RMSE": RMSE,
        "MAE": MAE,
        "R2": R2,
        "explained_variance": expvar,
        "quantiles": {
            "median_abs_error": float(np.percentile(flat_abs_err, 50)),
            "p95_abs_error": float(np.percentile(flat_abs_err, 95)),
        },
    })

    # ---------- optional energy norm ----------
    if K is not None:
        K = np.asarray(K)
        n_space = U.shape[-1]
        if K.shape != (n_space, n_space):
            raise ValueError(f"K must be shape (n_space, n_space)=({n_space},{n_space}). Got {K.shape}")

        # accumulate energy over samples (snapshot, time if present)
        # energy(err) = sum_i e_i^T K e_i ; energy(u) = sum_i u_i^T K u_i
        E_err = 0.0
        E_u = 0.0

        if not has_time:
            # iterate snapshots
            for i in range(U.shape[0]):
                e = err[i]
                uu = U[i]
                E_err += float(e @ (K @ e))
                E_u += float(uu @ (K @ uu))
        else:
            # iterate snapshot & time
            for i in range(U.shape[0]):
                for t in range(U.shape[1]):
                    e = err[i, t]
                    uu = U[i, t]
                    E_err += float(e @ (K @ e))
                    E_u += float(uu @ (K @ uu))

        metrics["energy_norm_error"] = float(np.sqrt(max(E_err, 0.0)))
        metrics["relative_energy_norm_error"] = float(
            metrics["energy_norm_error"] / np.maximum(np.sqrt(max(E_u, 0.0)), eps)
        )

    return metrics


def generate_rom_error_report(metrics, name="ROM Accuracy Report"):
    """
    Print a structured summary of ROM error metrics.

    Parameters
    ----------
    metrics : dict
        Error metrics dictionary from compute_rom_error_metrics_flat.
    name : str, optional
        Title of the report.
    """
    print("=" * len(name))
    print(name)
    print("=" * len(name))

    # --- Global Metrics ---
    print("\nGlobal Errors:")
    print(f"{'L2 Error:':<32} {metrics['L2_error']:.4e}")
    print(f"{'Relative L2 Error:':<32} {metrics['relative_L2_error']:.4e}")
    print(f"{'L∞ Error:':<32} {metrics['Linf_error']:.4e}")
    print(f"{'Relative L∞ Error:':<32} {metrics['relative_Linf_error']:.4e}")
    print(f"{'RMSE:':<32} {metrics['RMSE']:.4e}")
    print(f"{'MAE:':<32} {metrics['MAE']:.4e}")

    if "energy_norm_error" in metrics:
        print(f"{'Energy Norm Error:':<32} {metrics['energy_norm_error']:.4e}")
    if "relative_energy_norm_error" in metrics:
        print(f"{'Relative Energy Norm Error:':<32} {metrics['relative_energy_norm_error']:.4e}")

    # --- Statistical Fit Metrics ---
    print("\nStatistical Fit:")
    print(f"{'R² Score:':<32} {metrics['R2']:.4f}")
    print(f"{'Explained Variance:':<32} {metrics['explained_variance']:.4f}")

    # --- Error Distribution ---
    q = metrics["quantiles"]
    print("\nError Distribution (absolute):")
    print(f"{'Median Abs Error:':<32} {q['median_abs_error']:.4e}")
    print(f"{'95th Percentile Abs Error:':<32} {q['p95_abs_error']:.4e}")

    # --- Time-Dependent Errors ---
    print("\nTime/Parameter-Dependent Errors:")
    print(f"{'Average Relative L2 over samples:':<40} {metrics['time_avg_rel_L2_error']:.4e}")

    rel_t = metrics.get("relative_L2_error_time", None)
    if rel_t is not None:
        rel_flat = np.asarray(rel_t).reshape(-1)
        print(f"{'Max Relative L2 over samples:':<40} {np.max(rel_flat):.4e}")
        print(f"{'Min Relative L2 over samples:':<40} {np.min(rel_flat):.4e}")

    print("—" * 120)


def plot_rom_param_diagnostics_t(
    u, u_rom,
    *,
    l2_per_param=None,          # (p,) optional
    speedup_per_param=None,     # (p,) optional
    param_labels=None,          # length p optional
    eps=1e-16,
    log_error=True,
    use_raincloud=True,         # needs pandas + ptitprince
):
    """
    u, u_rom: (p, t, dof)

    Plots:
      1) per-parameter aggregated relative error (bar)
      2) per-parameter per-timestep error (boxplot)
      3) error vs speed-up (scatter) if speedup given
      4) L2 norms (bar) if l2 given
      5) raincloud: rel_err_p distribution (across parameters)
      6) raincloud: rel_err_pt grouped by parameter (across timesteps)
      7) raincloud: speedup distribution (across parameters) if speedup given
    """
    u = np.asarray(u)
    u_rom = np.asarray(u_rom)

    if u.shape != u_rom.shape or u.ndim != 3:
        raise ValueError(f"Expected u and u_rom as (p,t,dof) with same shape; got {u.shape} vs {u_rom.shape}")

    p, t, dof = u.shape

    if param_labels is None:
        param_labels = [f"p{i}" for i in range(p)]
    if len(param_labels) != p:
        raise ValueError(f"param_labels must have length p={p}; got {len(param_labels)}")

    if l2_per_param is not None:
        l2_per_param = np.asarray(l2_per_param).reshape(-1)
        if l2_per_param.size != p:
            raise ValueError(f"l2_per_param must have length p={p}; got {l2_per_param.size}")

    if speedup_per_param is not None:
        speedup_per_param = np.asarray(speedup_per_param).reshape(-1)
        if speedup_per_param.size != p:
            raise ValueError(f"speedup_per_param must have length p={p}; got {speedup_per_param.size}")

    diff = u - u_rom

    # per (p,t) relative error (L2 over dof)
    num_pt = np.linalg.norm(diff, axis=2)                 # (p,t)
    den_pt = np.linalg.norm(u, axis=2)                    # (p,t)
    rel_err_pt = num_pt / np.maximum(den_pt, eps)         # (p,t)

    # per p relative error (L2 over time+dof)
    num_p = np.linalg.norm(diff.reshape(p, -1), axis=1)   # (p,)
    den_p = np.linalg.norm(u.reshape(p, -1), axis=1)      # (p,)
    rel_err_p = num_p / np.maximum(den_p, eps)            # (p,)

    # ------------- helpers -------------
    def _maybe_log(v):
        v = np.asarray(v)
        return np.log10(np.maximum(v, eps)) if log_error else v

    def _raincloud_1d(values, xlabel, figsize=(8.7, 2.4)):
        import pandas as pd
        import ptitprince as pt

        values = np.asarray(values).reshape(-1)
        df = pd.DataFrame({"group": [""] * values.size, "val": values})

        fig, ax = plt.subplots(figsize=figsize)
        pt.RainCloud(
            x="group", y="val", data=df,
            bw=0.2,
            width_viol=1.0,
            orient="h",
            pointplot=False,
            dodge=False,
            alpha=1.0,
            width_box=0.30,
            linewidth=1,
            point_size=4.5,
            move=0.2,
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        fig.tight_layout()
        plt.show()

    # def _raincloud_grouped(groups, values, xlabel, figsize=(9.5, 4.0)):
    #     import pandas as pd
    #     import ptitprince as pt

    #     df = pd.DataFrame({"group": groups, "val": values})

    #     fig, ax = plt.subplots(figsize=figsize)
    #     pt.RainCloud(
    #         x="group", y="val", data=df,
    #         bw=0.2,
    #         width_viol=1.0,
    #         orient="v",
    #         pointplot=False,
    #         dodge=True,
    #         alpha=1.0,
    #         width_box=0.25,
    #         linewidth=1,
    #         point_size=2.5,
    #         move=0.15,
    #         ax=ax,
    #     )
    #     ax.set_xlabel("parameter")
    #     ax.set_ylabel(xlabel)
    #     ax.set_title("Raincloud grouped by parameter")
    #     fig.tight_layout()
    #     plt.show()

    # -------- 2) per-parameter distribution over time (boxplot) --------
    data = [rel_err_pt[i, :] for i in range(p)]
    data = [_maybe_log(d) for d in data]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, p + 1))
    ax.set_xticklabels(param_labels)
    ax.set_xlabel("parameter")
    ax.set_ylabel("log10(relative error per timestep)" if log_error else "relative error per timestep")
    ax.set_title("Error distribution over timesteps (by parameter)")
    fig.tight_layout()
    plt.show()

    # -------- 4) L2 norms (bar) --------
    if l2_per_param is not None:
        fig, ax = plt.subplots(figsize=(9.5, 3.2))
        ax.bar(np.arange(p), l2_per_param)
        ax.set_xticks(np.arange(p))
        ax.set_xticklabels(param_labels)
        ax.set_xlabel("parameter")
        ax.set_ylabel("% L2 norm")
        ax.set_title("% L2 error by parameter")
        fig.tight_layout()
        plt.show()

    # -------- 5-7) raincloud plots --------
    if use_raincloud:
        try:
            # 5) aggregated error across parameters
            _raincloud_1d(
                _maybe_log(rel_err_p),
                "log10(relative error) across parameters" if log_error else "relative error across parameters",
                figsize=(8.7, 2.4),
            )

            # 7) speed-up across parameters
            if speedup_per_param is not None:
                _raincloud_1d(speedup_per_param, "speed-up across parameters", figsize=(8.7, 2.2))

        except Exception as e:
            print(f"Raincloud unavailable ({type(e).__name__}: {e}). Fallback plots will be shown.")
            use_raincloud = False

    if not use_raincloud:
        # fallback for aggregated error
        fig, ax = plt.subplots(figsize=(8.7, 2.4))
        ax.hist(_maybe_log(rel_err_p), bins=30)
        ax.set_xlabel("log10(relative error) across parameters" if log_error else "relative error across parameters")
        ax.set_ylabel("count")
        fig.tight_layout()
        plt.show()

        # fallback for speed-up
        if speedup_per_param is not None:
            fig, ax = plt.subplots(figsize=(8.7, 2.2))
            ax.boxplot(speedup_per_param, vert=False)
            ax.set_xlabel("speed-up across parameters")
            ax.set_yticks([])
            fig.tight_layout()
            plt.show()

    return {
        "rel_err_p": rel_err_p,       # (p,)
        "rel_err_pt": rel_err_pt,     # (p,t)
        "l2_per_param": l2_per_param,
        "speedup_per_param": speedup_per_param,
    }
