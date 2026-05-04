"""ROM error metrics for flat data.

TL;DR
-----
This module computes, plots, and reports error measures between full-order and reduced-order reconstructions.

Notes
-----
The functions cover global statistics, per-snapshot diagnostics, optional energy norms, and simple report printing.
"""

import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd

def compute_rom_error_metrics_flat(u, u_rom, K=None):
    """compute_rom_error_metrics_flat
    
    TL;DR
    -----
    compute_rom_error_metrics_flat.
    
    Notes
    -----
    Compute various error metrics between full-order and ROM reconstructions for flat data.
    
    Parameters
    ----------
    u : array_like, shape (n_snapshots, n_space)
        Full-order field, with each row representing a snapshot.
    u_rom : array_like, shape (n_snapshots, n_space)
        ROM reconstruction matching the shape of `u`.
    K : array_like, shape (n_space, n_space), optional
        Stiffness matrix for computing the energy-norm error.
    
    Returns
    -------
    metrics : dict
        Dictionary containing error metrics:
        
        time-dependent
        ---------------
        L2_error_time : ndarray, shape (n_snapshots,)
            L2 norm of error per snapshot.
        relative_L2_error_time : ndarray, shape (n_snapshots,)
            Relative L2 error per snapshot.
        RMSE_time : ndarray, shape (n_snapshots,)
            Root mean square error per snapshot.
        MAE_time : ndarray, shape (n_snapshots,)
            Mean absolute error per snapshot.
        time_avg_rel_L2_error : float
            Average relative L2 error over all snapshots.
    
        global
        ------
        L2_error : float
            Global L2 norm of the error.
        relative_L2_error : float
            Global relative L2 error.
        Linf_error : float
            Maximum absolute error.
        relative_Linf_error : float
            Maximum relative error.
        RMSE : float
            Global root mean square error.
        MAE : float
            Global mean absolute error.
        R2 : float
            Coefficient of determination.
        explained_variance : float
            Variance explained by the ROM.
        quantiles : dict
            median_error : float
                Median absolute error.
            p95_error : float
                95th percentile of absolute errors.
    
        optional
        --------
        energy_norm_error : float
            Energy-norm error computed if `K` is provided.
    """
    N_sample = len(u)
    u   = np.array(u)
    u_r = np.array(u_rom)
    if u.shape != u_r.shape:
        raise ValueError("u and u_rom must have the same shape")
    # n_snap, n_space = u.shape
    err = u - u_r
    metrics = {}

    # --- time‐dependent ---
    # L2, rel L2, RMSE, MAE per snapshot
    L2_t    = np.linalg.norm(err, axis=1)/N_sample
    relL2_t = L2_t / np.linalg.norm(u, axis=1)
    RMSE_t  = np.sqrt(np.mean(err**2, axis=1))
    MAE_t   = np.mean(np.abs(err), axis=1)

    metrics.update({
      'L2_error_time':           L2_t,
      'relative_L2_error_time':  relL2_t,
      'RMSE_time':               RMSE_t,
      'MAE_time':                MAE_t,
      'time_avg_rel_L2_error':   np.mean(relL2_t),
    })

    # --- global ---
    flat_u   = u.ravel()
    flat_err = err.ravel()

    L2   = np.linalg.norm(flat_err)/N_sample
    relL2= L2 / (np.linalg.norm(flat_u)/N_sample)
    Linf = np.max(np.abs(flat_err))
    relLinf = Linf/np.max(np.abs(flat_u))
    RMSE = np.sqrt(np.mean(flat_err**2))
    MAE  = np.mean(np.abs(flat_err))

    ss_res = np.sum(flat_err**2)
    ss_tot = np.sum((flat_u - flat_u.mean())**2)
    R2     = 1 - ss_res/ss_tot
    expvar = np.var(u_r.ravel()) / np.var(flat_u)

    metrics.update({
      'L2_error':           L2,
      'relative_L2_error':  relL2,
      'Linf_error':         Linf,
      'relative_Linf_error':  relLinf,
      'RMSE':               RMSE,
      'MAE':                MAE,
      'R2':                 R2,
      'explained_variance': expvar,
      'quantiles': {
         'median_error': np.percentile(flat_err, 50),
         'p95_error':   np.percentile(np.abs(flat_err), 95)
      }
    })

    if K is not None:
        d = flat_err
        metrics['energy_norm_error'] = np.sqrt(d @ (K @ d))

    return metrics


def plot_rom_error_diagnostics_flat(u, u_rom, rom_relative_error, rom_speed_up, sim_axis, metrics, spatial_shape=None):
    """plot_rom_error_diagnostics_flat
    
    TL;DR
    -----
    plot_rom_error_diagnostics_flat.
    
    Notes
    -----
    Visualize ROM error diagnostics including scatter, spatial snapshots, and raincloud plots.
    
    Parameters
    ----------
    u : array_like, shape (n_snapshots, n_space)
        Full-order field for diagnostics.
    u_rom : array_like, shape (n_snapshots, n_space)
        ROM reconstruction matching shape of `u`.
    rom_relative_error : array_like, shape (n_snapshots,)
        Relative error per snapshot.
    rom_speed_up : array_like, shape (n_snapshots,)
        Speed-up factors per snapshot.
    sim_axis : tuple of str
        Axis labels for true vs ROM scatter (xlabel, ylabel).
    metrics : dict
        Dictionary of error metrics from compute_rom_error_metrics_flat.
    spatial_shape : tuple of int, optional
        Shape (nx, ny) to reshape spatial data for pcolormesh plots.
    
    Returns
    -------
    None
    """
    u   = np.array(u)
    u_r = np.array(u_rom)

    n_snap, *_ = u.shape

    # 1) scatter true vs. ROM
    fig, ax = plt.subplots()
    ax.scatter(u.ravel(), u_r.ravel(), s=1)
    ax.set_xlabel(sim_axis[0])
    ax.set_ylabel(sim_axis[1])
    ax.set_title( sim_axis[0] + ' vs. ' + sim_axis[1])
    ax.axis('equal')
    plt.show()
    
   
    # 2) spatial snapshots if shape provided
    if spatial_shape is not None:
        for idx in (0, n_snap//2, n_snap-1):
            for name, field in [('True', u), ('ROM', u_r), ('Error', u - u_r)]:
                snap = field[idx].reshape(spatial_shape)
                fig, ax = plt.subplots()
                pcm = ax.pcolormesh(snap, shading='auto')
                fig.colorbar(pcm, ax=ax)
                ax.set_title(f"{name} @ snapshot {idx}")
                ax.set_xlabel('x-index')
                ax.set_ylabel('y-index')
                plt.show()
    else:
        print("spatial_shape not given → skipping spatial snapshots.")

    df_B = pd.DataFrame({'':'','Relative error (log)%':np.log10(rom_relative_error)})
    # 
    fig, ax = plt.subplots(figsize=(8.7, 3))
    pt.RainCloud(x = '', y = 'Relative error (log)%', data = df_B, palette = "viridis", bw = 0.2,
        width_viol = 1., ax = ax, orient = "h",pointplot = False, dodge=False, alpha=1.0, width_box = 0.35, linewidth=1, point_size =6.0, move=0.2)
    
    # ===
    
    df_D = pd.DataFrame({'':'','Speed-up':rom_speed_up})

    fig, ax = plt.subplots(figsize=(8.6, 2))
    pt.RainCloud(x = '', y = 'Speed-up', data = df_D, palette = "plasma",
        width_viol = 1.0, ax = ax, orient = "h",pointplot = False, dodge=True, alpha=1.0, width_box = 0.25, linewidth=1, point_size =6.0)
    print("-" * 500)   # simple hyphens

def generate_rom_error_report(metrics, name="ROM Accuracy Report"):
    """generate_rom_error_report
    
    TL;DR
    -----
    generate_rom_error_report.
    
    Notes
    -----
    Print a structured summary of ROM error metrics.
    
    Parameters
    ----------
    metrics : dict
        Error metrics dictionary from compute_rom_error_metrics_flat.
    name : str, optional
        Title of the report. Defaults to "ROM Accuracy Report".
    
    Returns
    -------
    None
    """
    print("="*len(name))
    print(name)
    print("="*len(name))

    # --- Global Metrics ---
    print("\nGlobal Errors:")
    print(f"{'L2 Error:':<25} {metrics['L2_error']:.4e}")
    print(f"{'Relative L2 Error:':<25} {metrics['relative_L2_error']:.4e}")
    print(f"{'L∞ Error:':<25} {metrics['Linf_error']:.4e}")
    print(f"{'Relative L∞ Error:':<25} {metrics['relative_Linf_error']:.4e}")

    print(f"{'RMSE:':<25} {metrics['RMSE']:.4e}")
    print(f"{'MAE:':<25} {metrics['MAE']:.4e}")

    if 'energy_norm_error' in metrics:
        print(f"{'Energy Norm Error:':<25} {metrics['energy_norm_error']:.4e}")

    # --- Statistical Fit Metrics ---
    print("\nStatistical Fit:")
    print(f"{'R² Score:':<25} {metrics['R2']:.4f}")
    print(f"{'Explained Variance:':<25} {metrics['explained_variance']:.4f}")

    # --- Error Distribution ---
    q = metrics['quantiles']
    print("\nError Distribution:")
    print(f"{'Median Error:':<25} {q['median_error']:.4e}")
    print(f"{'95th Percentile Error:':<25} {q['p95_error']:.4e}")

    # --- Time-Dependent Errors ---
    print("\nTime/Parameter-Dependent Errors:")
    print(f"{'Average Rel L2 Error over time/parameter:':<35} {metrics['time_avg_rel_L2_error']:.4e}")
    print(f"{'Max Rel L2 Error over time/parameter:':<35} {np.max(metrics['relative_L2_error_time']):.4e}")
    print(f"{'Min Rel L2 Error over time/parameter:':<35} {np.min(metrics['relative_L2_error_time']):.4e}")
    print("―" * 500)   # — EM DASH × 40, or…


