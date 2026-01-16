import numpy as np
import shutil
from pathlib import Path
import os


def visualize2D(u,basis):
    from skfem.visuals.matplotlib import plot, show
    return plot(basis,
                u,
                shading='gouraud',
                colorbar='True').show()


def save_vtk_solution(u, mesh, basis, scale, run_dir, prefix, split_dim=False):
    """
    Write a single solution vector to a VTK file.

    Applies a translation to the mesh based on displacement values and saves
    the resulting geometry along with point data fields.

    Parameters
    ----------
    u : array_like
        Displacement vector of length matching the mesh degrees of freedom.
    mesh : object
        Mesh object supporting `translated(displacements)` to return a new mesh
        and `save(path, point_data=...)` to write VTK files.
    basis : object
        Basis object containing attribute `nodal_dofs`, an integer array indexing
        into the global solution vector for nodal degrees of freedom.
    scale : float
        Scalar multiplier applied to the displacement values before translation.
    run_dir : pathlib.Path
        Directory in which the `.vtk` file will be created.
    prefix : str
        Filename prefix (e.g., "test_sol_fos").
    split_dim : bool
        If True, splits the displacement into separate scalar fields
        (`u_x`, `u_y`, and `u_z` for 3D) in the VTK output; otherwise
        writes a single vector field `u`.

    Returns
    -------
    None

    [Author: Suparno Bhattacharyya]
    """
    # extract and scale only the nodal displacements
    u_node = scale * u[basis.nodal_dofs]     # shape (dim, n_nodes)

    # apply displacement and get translated mesh
    m = mesh.translated(u_node)

    filename = run_dir / f"{prefix}.vtk"


    if split_dim:
        # prepare the point_data dict
        point_data = {}

        if u_node.ndim == 1:
            # fallback: scalar field
            point_data["u"] = u_node
        else:
            # vector field case
            dim, n_nodes = u_node.shape
            # when split_dim, emit separate scalars
            if split_dim:
                labels = ["x", "y", "z"][:dim]
                for d in range(dim):
                    point_data[f"u_{labels[d]}"] = u_node[d, :]
            else:
                # assemble a (n_nodes, dim) array for VTK vector field
                vec = u_node.T  # now shape (n_nodes, dim)
                point_data["u"] = vec

        # write out
        m.save(str(filename), point_data=point_data)

    else:
        m.save(str(filename))



def generate_vtk(
    LS_test,
    LS_rom,
    mesh,
    basis,
    scale: float = 1.0,
    num_test: int = 5,
    out_dir: str = "sol_vtk_files",
    split_dim: bool = False
):
    """
    Batch export of full-order and reduced-order solutions to VTK.

    Randomly selects solution indices, generates translated meshes, and
    writes both full-order (FOS) and reduced-order (ROM) displacement fields
    to VTK files within separate test directories. Cleans output directory on
    each invocation.

    Parameters
    ----------
    LS_test : sequence of array_like
        List or array of full-order solution vectors.
    LS_rom : sequence of array_like
        List or array of reduced-order solution vectors corresponding to `LS_test` indices.
    mesh : object
        Mesh object used for geometry translations (see `_save_vtk_solution`).
    basis : object
        Basis object with attribute `nodal_dofs` for nodal indexing.
    scale : float, optional
        Scale factor for displacements before applying to the mesh (default is 1.0).
    num_test : int, optional
        Number of random test cases to export (default is 5).
    out_dir : str, optional
        Base directory path where subdirectories `Test_1`, `Test_2`, … will be created
        (default is "sol_vtk_files").
    split_dim : bool, optional
        If True, split displacement into per-axis scalar fields in VTK outputs
        (default is False).

    Returns
    -------
    None

    Notes
    -----
    - If `out_dir` already exists, it will be removed entirely before new output
      is written.
    - Each `Test_i` directory contains two files:
      `test_sol_fos_i.vtk` and `test_sol_rom_i.vtk`.

    Examples
    --------
    >>> generate_vtk(LS_test, LS_rom, mesh, basis, scale=0.5, num_test=3,
    ...              out_dir="vtk_outputs", split_dim=True)

    [Author: Suparno Bhattacharyya]
    """
    base_dir = Path(out_dir)
    # if base_dir.exists():
    #     shutil.rmtree(base_dir)
    # base_dir.mkdir()
    os.makedirs(out_dir, exist_ok=True)


    for i in range(1, num_test + 1):
        idx = np.random.randint(len(LS_test))
        run_dir = base_dir / f"Test_{i}"

        # if run_dir.exists():
        #     shutil.rmtree(run_dir)
        # run_dir.mkdir()
        os.makedirs(run_dir, exist_ok=True)

        # write full-order solution
        save_vtk_solution(
            LS_test[idx], mesh, basis, scale,
            run_dir, prefix=f"test_sol_fos_{i}", split_dim=split_dim
        )
        # write reduced-order solution
        save_vtk_solution(
            LS_rom[idx], mesh, basis, scale,
            run_dir, prefix=f"test_sol_rom_{i}", split_dim=split_dim
        )

def save_vtk_time_series(
    U: np.ndarray,                # shape = (ndofs, n_steps)
    times: np.ndarray,            # shape = (n_steps,)
    mesh,
    basis,
    scale: float,
    run_dir: Path,
    prefix: str
):
    """
    Write one VTK per time-step and a .pvd that collects them.

    U      : full displacement history
    times  : time vector
    mesh   : skfem mesh
    basis  : skfem basis
    scale  : displacement scale
    run_dir: output directory
    prefix : file prefix, e.g. "beam"

    [Author: Suparno Bhattacharyya]
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    pvd_file = run_dir / f"{prefix}.pvd"

    # start PVD file
    with pvd_file.open("w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        U_plot = U[:, ::10]
        for i, t in enumerate(times[::10]):
            u = U_plot[:, i]
            frame = f"{prefix}_{i:04d}"
            # write one .vtk for this step
            save_vtk_solution(u, mesh, basis, scale, run_dir, frame)
            # add entry to .pvd
            f.write(f'    <DataSet timestep="{t}" group="" part="0" file="{frame}.vtk"/>\n')

        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')



def save_vtk_time_series_point(
    U: np.ndarray,                # shape = (ndofs, n_steps)
    mesh,
    run_dir: Path,
    prefix: str,
    interval: int = 10,
    point_data_name: str = "Temperature"
):
    """
    Write one VTK per time-step and a .pvd that collects them.

    U      : full displacement history
    mesh   : skfem mesh
    prefix : file prefix, e.g. "beam"
    interval : step interval for saving
    point_data_name : name of the point data field in VTK

    [Author: Suparno Bhattacharyya]
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, u in enumerate(U[::interval,:]):
        frame = f"{prefix}_{i:04d}"
        # write one .vtk for this step
        mesh.save(run_dir / f"{frame}.vtk", point_data={point_data_name: u})