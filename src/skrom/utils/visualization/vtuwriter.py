import os
import xml.etree.ElementTree as ET
from pathlib import Path
import meshio

"""
VTU Series Writer

Utilities to write time-series VTU files and an accompanying PVD collection
for visualization in ParaView. Supports both meshio.Mesh and SciKit-FEM MeshTet1 inputs.
"""

class VTUSeriesWriter:
    """
    Collect and export simulation snapshots as VTU and PVD files.

    Manages writing of individual VTU files at specified time steps and
    generates a PVD index file for seamless time-series playback.
    """
    def __init__(self, mesh, output_dir, *, prefix="step", skip=2, cell_type="tetra"):
        """
        Initialize the VTU series writer.

        Converts a SciKit-FEM mesh to meshio.Mesh if necessary and sets up
        output directory and naming conventions for subsequent VTU exports.

        Parameters
        ----------
        mesh : meshio.Mesh or MeshTet1
            Mesh input for exporting; SciKit-FEM MeshTet1 meshes are converted
            to meshio.Mesh internally.
        output_dir : str or Path
            Directory path where VTU and PVD files will be written. Created if needed.
        prefix : str, optional
            Filename prefix for VTU files (default is "step").
        skip : int, optional
            Interval of time steps between writes; only every `skip`-th step is saved
            (default is 2).
        cell_type : str, optional
            Cell type label for meshio (e.g., "tetra", "triangle") when converting
            from SciKit-FEM MeshTet1 (default is "tetra").

        Raises
        ------
        TypeError
            If `mesh` is not a meshio.Mesh and lacks attributes `p` and `t` for conversion.

        Examples
        --------
        >>> writer = VTUSeriesWriter(mesh, "output/vtu", prefix="temp", skip=5)

        [Author: Suparno Bhattacharyya]
        """
        # Convert SciKit-FEM mesh to meshio.Mesh if necessary
        if not isinstance(mesh, meshio.Mesh):
            if hasattr(mesh, "p") and hasattr(mesh, "t"):
                pts = mesh.p.T  # (N, 3)
                conn = mesh.t.T  # (n_cells, 4)
                mesh = meshio.Mesh(
                    points=pts,
                    cells=[(cell_type, conn)]
                )
            else:
                raise TypeError("VTUSeriesWriter needs a meshio.Mesh or MeshTet1")
        
        self.mesh_points = mesh.points
        self.mesh_cells = mesh.cells
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.skip = skip
        self.entries = []

    def write_step(self, u, t, idx):
        """
        Write a VTU file for a simulation snapshot.

        Creates a meshio.Mesh with updated point_data and writes it to disk if
        the snapshot index matches the skip interval.

        Parameters
        ----------
        u : array_like
            Point-wise scalar data array (e.g., temperature) of length equal to
            the number of mesh points.
        t : float or int
            Simulation time corresponding to this snapshot.
        idx : int
            Snapshot index; only written if `idx % skip == 0`.

        Returns
        -------
        None

        Examples
        --------
        >>> writer.write_step(temp_array, time, step_index)

        [Author: Suparno Bhattacharyya]
        """
        if idx % self.skip != 0:
            return
        
        current_mesh = meshio.Mesh(
            points=self.mesh_points,
            cells=self.mesh_cells,
            point_data={"Temperature": u}
        )
        
        filename = f"{self.prefix}_{idx:03d}.vtu"
        current_mesh.write(self.out / filename, compression="zlib")
        self.entries.append((t, filename))

    def write_pvd(self, pvd_name="collection.pvd"):
        """
        Generate a PVD collection file for all written VTU snapshots.

        Iterates over recorded entries and constructs an XML-based PVD file
        that ParaView can use to load time-series data.

        Parameters
        ----------
        pvd_name : str, optional
            Filename for the PVD output (default is "collection.pvd").

        Returns
        -------
        None

        Examples
        --------
        >>> writer.write_pvd("simulation.pvd")

        [Author: Suparno Bhattacharyya]
        """
        root = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian"
        )
        collection = ET.SubElement(root, "Collection")
        
        for timestep, vtk_file in self.entries:
            ET.SubElement(
                collection, "DataSet",
                timestep=str(timestep),
                group="",
                part="0",
                file=vtk_file
            )
        
        tree = ET.ElementTree(root)
        tree.write(self.out / pvd_name, encoding="utf-8", xml_declaration=True)