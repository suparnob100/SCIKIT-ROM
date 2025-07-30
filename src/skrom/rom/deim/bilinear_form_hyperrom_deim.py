"""
hyperreduce/bilinear_form_hyperrom.py
------------------------------------
Implements Hyper-Reduction (HYPERROM) for reduced-order stiffness assembly.

This module provides:
  - `BilinearFormHYPERROM`: a subclass of `skfem.assembly.form.bilinear_form.BilinearForm`
    that
      * clusters elements by number of free DOFs after Dirichlet condensation
      * extracts and projects element stiffness blocks onto test/trial reduced bases
      * assembles the global reduced stiffness matrix via vectorized contractions

The `hyperreduce` folder contains all tools for hyper-reduction, including:
  - Classes for reduced‐order bilinear and linear forms with element clustering
  - Routines to extract local element matrices/vectors in the ROM basis
  - Utilities for efficient handling of Dirichlet conditions in reduced spaces
  - Support for element‐wise parallelization and weighted assembly
"""

from typing import Optional
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.bilinear_form import BilinearForm  
from skfem.assembly.basis import AbstractBasis
from numpy.typing import DTypeLike 
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

class BilinearFormHYPERROM_deim(BilinearForm):
    """
    Reduced-order bilinear form with element-weighted hyper-reduction.
    Adds a DEIM path: assemble only sampled rows (P^T J) and lift with deim_mat.
    """

    def __init__(self, form, elem_weight,
                 ubasis: Basis, lob, rob,
                 sampled_rows, deim_mat,
                 vbasis: Optional[Basis] = None,
                 free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None,
                 nthreads: int = 0,
                 dtype: DTypeLike = np.float64):

        super().__init__(form)

        # ---------------- core variables ----------------
        self.rob = rob                  # right/trial basis (N_free × r)

        # ------------- weights / sampled mesh --------
        self.weight           = np.array(elem_weight)
        self.nonzero_elements = np.nonzero(self.weight)[0]
        self.ubasis_rom       = ubasis.with_elements(self.nonzero_elements)

        # --------- DEIM info ----------
        self.sampled_rows      = np.asarray(sampled_rows, dtype=int)      # len = n_samp
        self.n_samp            = self.sampled_rows.size
        self.deim_mat = deim_mat   # (r × n_samp)

        ### element‐DOF mapping: shape (n_elems, n_loc)

        self.edofs = self.ubasis_rom.element_dofs.T
        self.n_elems, self.n_loc = self.edofs.shape
        self.n_dofs = int(self.ubasis_rom.nodal_dofs.max()) + 1

        # Broadcast DOFs to the same shape as em for row/col indices
        # rows[i,a,b] = edofs[i,a], cols[i,a,b] = edofs[i,b]

        em_shape = tuple((self.edofs.shape[0],self.edofs.shape[1],self.edofs.shape[1]))
        self.rows = np.broadcast_to(self.edofs[:, :, None], em_shape)
        self.cols = np.broadcast_to(self.edofs[:, None, :], em_shape)

        self.row_flat  = self.rows.ravel()
        self.col_flat  = self.cols.ravel()


    def assemble_deim(self, **kwargs):
        """
        Assemble the globally weighted reduced stiffness matrix.

        Each element stiffness block is weighted, projected onto reduced
        test/trial bases restricted to free DOFs, and summed into a
        reduced r-by-r matrix.

        Parameters
        ----------
        **kwargs
            Additional options passed to the low-level `form` assembly
            routines (e.g., quadrature settings).

        Returns 
        -------
        K_reduced : ndarray, shape (r, r)
            Assembled reduced stiffness matrix.
        """

        Sampled_assembly = self.deim_elem_assembly(**kwargs)
        Reduced_matrix = self.deim_mat @ Sampled_assembly[self.sampled_rows] @ self.rob

        return Reduced_matrix
    

    def deim_elem_assembly(self, **kwargs):
        # extract element‐wise stiffness tensors: shape (n_elems, n_loc, n_loc)
        em = self.extract_element_matrices_rom(
            self.ubasis_rom, self.ubasis_rom,
            elem_indices=self.nonzero_elements,
            **kwargs
        )

        # Flatten everything
        data_flat = em.ravel()

        # (Optional) drop zero entries to save memory
        nz = data_flat != 0
        data_flat = data_flat[nz]
        row_flat  = self.row_flat[nz]
        col_flat  = self.col_flat[nz]

        # Build sparse matrix in one shot and convert to CSR
        K = coo_matrix(
            (data_flat, (row_flat, col_flat)),
            shape=(self.n_dofs, self.n_dofs)
        ).tocsr()

        return K


    def extract_element_matrices_rom(self, ubasis: Basis,
                                     vbasis: Optional[Basis] = None,
                                     elem_indices: Optional[ndarray] = None,
                                     **kwargs) -> ndarray:
        """
        Extract local stiffness matrices in the reduced basis for specified elements.

        This routine assembles the original bilinear form on each element
        and returns an array of shape (n_elems, Nbfun, Nbfun), where Nbfun
        is the number of local basis functions.

        Parameters
        ----------
        ubasis : Basis
            Trial-space finite element basis (with restricted elements if
            `elem_indices` is provided).
        vbasis : Basis, optional
            Test-space finite element basis; defaults to `ubasis`.
        elem_indices : ndarray of int, optional
            Subset of element indices to restrict the basis via `with_elements`.
        **kwargs
            Extra keyword arguments forwarded to the form assembly.

        Returns
        -------
        element_matrices : ndarray, shape (n_elems, Nbfun, Nbfun)
            Local element stiffness matrices for each (restricted) element.

        Raises
        ------
        ValueError
            If trial/test bases have mismatched quadrature dimensions.
        """
        if vbasis is None:
            vbasis = ubasis
        elif ubasis.X.shape[-1] != vbasis.X.shape[-1]:
            raise ValueError("Quadrature mismatch: trial and test functions should have the same number of integration points.")

        # Now, all data (including dx, element_dofs, etc.) refer only to the restricted set.
        nt = ubasis.nelems         # Number of (restricted) elements
        dx = ubasis.dx             # Quadrature weights per element (already restricted)

        # Combine default parameters with any additional keyword arguments.
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })

        wdict['elem_indices'] = elem_indices

        # Its shape will be (Nbfun, Nbfun, nt)
        local_data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=self.dtype)
        # Serial computation if no threading is requested.


        if self.nthreads <= 0:
            for j in range(ubasis.Nbfun):
                for i in range(vbasis.Nbfun):
                    local_data[j, i, :] = self._kernel(
                        ubasis.basis[j],
                        vbasis.basis[i],
                        wdict,
                        dx
                    )
        else:
            # Prepare index pairs for threaded computation.
            from itertools import product
            indices = np.array([[i, j] for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))])
            threads = [
                Thread(
                    target=self._threaded_kernel,
                    args=(local_data, ix, ubasis.basis, vbasis.basis, wdict, dx)
                )
                for ix in np.array_split(indices, self.nthreads, axis=0)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Rearrange data from shape (Nbfun, Nbfun, n_elements) to (n_elements, Nbfun, Nbfun)
        element_matrices = local_data.transpose(2, 0, 1)
        return element_matrices

