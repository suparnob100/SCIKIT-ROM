from pathlib import Path
import json
import numpy as np
from .vtuwriter import VTUSeriesWriter
"""
VTU Series Conversion Utilities

This module provides functions to construct SciKit-FEM meshes from parameter files
and to convert NumPy solution arrays into VTU series for post-processing and
visualization. It scans directories for solution data, rebuilds meshes, writes
VTU snapshots at specified strides, and aggregates them into PVD collection files.

[Author: Suparno Bhattacharyya]
"""


def save_vtu_time_series_point(
    U: np.ndarray,
    mesh,
    run_dir: Path,
    prefix: str,
    interval: int = 10,
    point_data_name: str = "Temperature",
):
    """
    Write point-data frames to VTK on an undeformed mesh.

    The function writes frames named ``{prefix}_{k:04d}.vtk`` with a point-data
    array stored under ``point_data_name``.

    The time axis handling is:
    - if ``U.shape[0] == n_points``: treat U as (n_points, n_steps) and save columns
    - else: treat U as (n_steps, n_points) and save rows

    Parameters
    ----------
    U : numpy.ndarray
        Point-data history with shape (n_steps, n_points) or (n_points, n_steps).
    mesh : object
        Mesh object that provides ``save(path, point_data=...)``.
    run_dir : pathlib.Path
        Output directory.
    prefix : str
        Base name used for VTU frames.
    interval : int, optional
        Save every ``interval`` steps. Default is 10.
    point_data_name : str, optional
        Key used in VTK point_data. Default is "Temperature".

    Returns
    -------
    None

    Notes
    -----
    Authors: Suparno Bhattacharyya
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    U = np.asarray(U)
    n_points = mesh.p.shape[1] if hasattr(mesh, "p") else None

    if n_points is not None and U.ndim == 2 and U.shape[0] == n_points:
        # U: (n_points, n_steps)
        step_ids = range(0, U.shape[1], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: U[:, step]})
    else:
        # U: (n_steps, n_points)
        step_ids = range(0, U.shape[0], interval)
        for out_i, step in enumerate(step_ids):
            frame = f"{prefix}_{out_i:04d}"
            mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: U[step, :]})


def save_vtu_time_series_point_vertexsample(U, mesh, basis, run_dir, prefix, interval=8, point_data_name="Temperature"):
    """
    U: (n_steps, ndofs_P2)
    Saves vertex-sampled field (n_vertices,) on the original mesh.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    P = basis.probes(mesh.p)                     # maps P2 dofs -> values at vertices (n_vertices x ndofs)

    step_ids = np.arange(0, U.shape[0], interval)
    for out_i, step in enumerate(step_ids):
        frame = f"{prefix}_{out_i:04d}"
        u_vertex = P @ U[step, :]                # (n_vertices,)
        mesh.save(run_dir / f"{frame}.vtu", point_data={point_data_name: u_vertex})


def save_vtu_point_vertexsample(U, mesh, basis, run_dir, fname, point_data_name="Temperature"):
    """
    U: (n_steps, ndofs_P2)
    Saves vertex-sampled field (n_vertices,) on the original mesh.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    P = basis.probes(mesh.p)                     # maps P2 dofs -> values at vertices (n_vertices x ndofs)

    u_vertex = P @ U                # (n_vertices,)

    mesh.save(run_dir / f"{fname}.vtu", point_data={point_data_name: u_vertex})