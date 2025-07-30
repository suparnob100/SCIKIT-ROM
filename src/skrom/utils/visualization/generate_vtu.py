from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from .vtuwriter import VTUSeriesWriter
from .domain import domain

"""
VTU Series Conversion Utilities

This module provides functions to construct SciKit-FEM meshes from parameter files
and to convert NumPy solution arrays into VTU series for post-processing and
visualization. It scans directories for solution data, rebuilds meshes, writes
VTU snapshots at specified strides, and aggregates them into PVD collection files.

[Author: Suparno Bhattacharyya]
"""

def build_mesh_from_params(p: dict):
    """
    Construct a SciKit-FEM mesh from JSON parameter entries.

    Reads domain dimensions and mesh refinement factor from a parameter dictionary
    and uses them to build a finite-element mesh via the `domain` factory.

    Parameters
    ----------
    p : dict
        Dictionary of mesh parameters. Expected keys (3D):
        - ``lx`` : float, optional
            Domain length in the x-direction (default is 1.0).
        - ``ly`` : float, optional
            Domain length in the y-direction (default is 1.0).
        - ``lz`` : float, optional
            Domain length in the z-direction (default is 1.0).
        - ``factor`` : int, optional
            Mesh refinement factor controlling element subdivision (default is 4).

    Returns
    -------
    mesh : Mesh
        A SciKit-FEM mesh object constructed with the specified dimensions
        and refinement factor.

    Examples
    --------
    >>> params = {'lx': 2.0, 'ly': 1.0, 'lz': 0.5, 'factor': 6}
    >>> mesh = build_mesh_from_params(params)
    >>> mesh.p.shape  # number of spatial dimensions and nodes
    (3, N)

    Notes
    -----
    - If any of the dimension keys are missing, defaults of 1.0 each are used.
    - `factor` must be convertible to int; non-integer inputs will be cast.
    """
    lx = p.get("lx", 1.0)
    ly = p.get("ly", 1.0)
    lz = p.get("lz", 1.0)
    factor = int(p.get("factor", 4))
    mesh, *_ = domain(lx, ly, lz, factor=factor)
    return mesh


def convert_to_vtu_series(
    root_dir: str | Path,
    sol_file_name: str = "u_solution.npy",
    vtu_folder_name: str = "VTU",
    steps: int = 300,
    stride: int = 10
) -> int:
    """
    Convert NumPy solution snapshots to a VTU series with PVD index.

    Scans all subdirectories under `root_dir` for pairs of ``params.json`` and
    solution files, rebuilds the corresponding mesh, writes VTU files for each
    snapshot at intervals defined by `stride` up to `steps`, and aggregates
    outputs into a PVD file for streamlined visualization.

    Parameters
    ----------
    root_dir : str or Path
        Base directory to search recursively for solution runs.
    sol_file_name : str, optional
        Filename of the NumPy solution array (default is ``"u_solution.npy"``).
    vtu_folder_name : str, optional
        Name of the subfolder to create for VTU outputs in each run directory
        (default is ``"VTU"``).
    steps : int, optional
        Maximum number of time steps to process from each solution array
        (default is 300).
    stride : int, optional
        Interval between snapshots to write (every `stride` steps)
        (default is 10).

    Returns
    -------
    processed : int
        Total number of run folders processed (i.e., those containing both
        ``params.json`` and the solution file).

    Raises
    ------
    IOError
        If reading ``params.json`` or the NumPy array fails for a detected folder.

    Examples
    --------
    >>> n = convert_to_vtu_series(
    ...     'sim_runs', sol_file_name='solutions/u.npy',
    ...     vtu_folder_name='VTU_out', steps=200, stride=5
    ... )
    >>> print(f"Processed {n} runs.")

    Notes
    -----
    - Existing ``root_dir`` contents are not modified or deleted; new VTU folders
      are created alongside original data.
    - Uses `tqdm` for a progress bar when scanning directories.
    - PVD writer organizes all snapshot VTU files for each run into a single
      index file for use with ParaView or similar tools.
    """
    root = Path(root_dir).expanduser().resolve()
    processed = 0

    for folder in tqdm(sorted(root.glob("**/"))):
        par = folder / "params.json"
        sol = folder / sol_file_name
        if not par.exists() or not sol.exists():
            continue

        params = json.loads(par.read_text())
        u_sol = np.load(sol)
        mesh = build_mesh_from_params(params)

        writer = VTUSeriesWriter(
            mesh=mesh,
            output_dir=folder / vtu_folder_name,
            prefix="skfem",
            skip=stride
        )

        for step in range(min(steps, len(u_sol))):
            if step % stride == 0:
                writer.write_step(u_sol[step], step, step)

        writer.write_pvd()
        processed += 1

    return processed
