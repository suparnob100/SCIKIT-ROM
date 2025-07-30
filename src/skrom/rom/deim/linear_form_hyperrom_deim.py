"""
hyperreduce/linear_form_hyperrom.py
-----------------------------------
Implements Hyper-Reduction (HYPERROM) for reduced-order load vector assembly.

This module provides:
  - `LinearFormHYPERROM`: a subclass of `skfem.assembly.form.linear_form.LinearForm`
    that projects element-wise load contributions onto a reduced basis, clusters
    elements by free-DOF count after Dirichlet condensation, and assembles the
    global reduced load vector via vectorized weighted projections.

The `hyperreduce` folder contains all tools to perform hyper-reduction, including:
  - Reduced-order bilinear forms (`BilinearFormHYPERROM`) and linear forms
    (`LinearFormHYPERROM`)
  - Routines for extracting element stiffness matrices and load vectors in a
    reduced basis
  - Utilities for efficient handling of Dirichlet conditions and element clustering
  - Support for weights, parallelization, and reconstruction of full-order data
"""
from typing import Optional
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.linear_form import LinearForm  # Import the full-order linear form class
from skfem.assembly.basis import AbstractBasis
from numpy.typing import DTypeLike 

class LinearFormHYPERROM_deim(LinearForm):
    def __init__(self, form, elem_weight, ubasis: Basis, lob,
                 sampled_rows, deim_mat,
                 free_dofs: Optional[np.ndarray] = None,
                 mean: Optional[np.ndarray] = None,
                 nthreads=0, dtype=np.float64):

        super().__init__(form)

        # ---------------- core members ----------------
        self.r_basis   = lob                  # (N_free, r)

        # ------------- weights / element subset --------
        self.weight           = np.array(elem_weight)
        self.nonzero_elements = np.nonzero(self.weight)[0]
        self.ubasis           = ubasis
        self.ubasis_rom       = ubasis.with_elements(self.nonzero_elements)

        # --------- DEIM sets (rows) ----------
        self.sampled_rows      = np.asarray(sampled_rows, dtype=int)     # len = n_samp
        self.n_samp            = self.sampled_rows.size
        self.deim_mat = deim_mat  # (r, n_samp)

        ###
        self.edofs = self.ubasis_rom.element_dofs   # (n_elems, 2)
        self.n_dofs = self.ubasis_rom.nodal_dofs.max()+1
        self.rows = self.edofs.ravel()                   # length = n_elems * n_loc

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
        # Combine default parameters with any additional keyword arguments.
        wdict = FormExtraParams({
            **self.ubasis_rom.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, self.ubasis_rom),
        })

        wdict['elem_indices'] = self.nonzero_elements

        Sampled_assembly = self.deim_elem_assembly(**kwargs)
        Reduced_vector = self.deim_mat @ Sampled_assembly[self.sampled_rows]

        return Reduced_vector
    

    def deim_elem_assembly(self, **kwargs):

        element_vectors = self.extract_element_vector_rom(self.ubasis_rom, elem_indices=self.nonzero_elements, **kwargs)
        data = element_vectors.T.ravel()       # same length, in matching order

        # add contributions at once
        f = np.zeros(self.n_dofs)
        np.add.at(f, self.rows, data)

        return f


    def extract_element_vector_rom(self, basis: Basis, elem_indices: Optional[np.ndarray] = None, **kwargs):
        """
        Extract local element load vectors in the reduced setting.

        Evaluates the original linear form on each specified element and returns
        an array of shape (n_elem, Nbfun), where Nbfun is the number of local
        basis functions per element.

        Parameters
        ----------
        basis : Basis
            Basis restricted via `with_elements` for trial functions.
        elem_indices : ndarray of int, optional
            Subset of elements to include; passed to `with_elements`.
        **kwargs
            Extra keyword arguments forwarded to low-level form evaluation.

        Returns
        -------
        element_vectors : ndarray, shape (n_elem, Nbfun)
            Local load vectors for each (restricted) element.

        Raises
        ------
        ValueError
            If `basis` is None or improperly configured.
        """
       
        # Fallback: ensure basis is defined.
        if basis is None:
            raise ValueError("A valid basis must be provided.")

        nt = basis.nelems  # number of (restricted) elements
        dx = basis.dx      # quadrature weights per element (already restricted)

        # Build the extra parameters dictionary.
        wdict = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        wdict['elem_indices'] = elem_indices

        # Allocate an array for local load vectors.
        # local_data has shape (Nbfun, nt)
        local_data = np.zeros((basis.Nbfun, nt), dtype=self.dtype)

        # Serial computation if no threading is requested.
        if self.nthreads <= 0:
            for i in range(basis.Nbfun):
                local_data[i, :] = self._kernel(basis.basis[i], wdict, dx)
        else:
            # Prepare indices for threaded computation.
            indices = np.arange(basis.Nbfun)
            # Define a helper for threaded computation.
            def threaded_kernel_vector(data, idx_chunk, basis_list, wdict, dx):
                for i in idx_chunk:
                    data[i, :] = self._kernel(basis_list[i], wdict, dx)
            idx_chunks = np.array_split(indices, self.nthreads)
            threads = [
                Thread(target=threaded_kernel_vector,
                       args=(local_data, chunk, basis.basis, wdict, dx))
                for chunk in idx_chunks
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Rearrange data from shape (Nbfun, nt) to (nt, Nbfun)
        element_vectors = local_data.transpose(1, 0)

        return element_vectors
  