"""
Implements reduced-order linear form assembly for full-order to reduced-order transformations.

This module provides:
  - `LinearFormROM`: a subclass of `skfem.assembly.form.linear_form.LinearForm`
    that projects full-order element load vectors onto reduced bases,
    groups elements by Dirichlet-free and mixed-Dirichlet sets for memory-efficient handling,
    and assembles the global reduced load vector with optional chunked computation.

The `rom` folder contains core tools for reduced-order modeling (ROM), including:
  - Classes for projecting and assembling reduced-order bilinear and linear forms
  - Utilities for handling Dirichlet boundary conditions in reduced spaces
  - Chunked and clustered assembly routines to manage large-scale stiffness/load data
  - Mapping utilities between full-order and reduced-order degrees of freedom

  [Author: Suparno Bhattacharyya]
"""
import gc
from typing import Optional
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form import Form
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.linear_form import LinearForm  # Import the full-order linear form class
from skfem.assembly.basis import AbstractBasis

class LinearFormROM(LinearForm):

    """
    LinearFormROM

    Linear form that projects element load vectors onto reduced bases
    and assembles the global reduced load vector, handling Dirichlet boundary
    conditions via mappings from full to free DOFs.

    Attributes
    ----------
    r_basis : ndarray, shape (N_free, r) or (N, r)
        Reduced basis for load vectors.
    free_dofs : ndarray or None
        Indices of global free (non-Dirichlet) DOFs.
    mean : ndarray or None
        Mean snapshot vector subtracted before basis computation.
    nthreads : int
        Number of threads for parallel computation.
    dtype : data-type
        Numeric type for computations.
    ubasis : Basis
        Full-order finite element basis for test functions.
    mapping : ndarray, shape (N_full,)
        Maps global DOF indices to reduced DOF indices or -1 for Dirichlet DOFs.
    element_dofs : ndarray
        Local-to-global DOF mapping for each element.
    free_indices : ndarray
        Reduced DOF indices for each element and basis function.
    mask : ndarray of bool
        Indicates free DOFs per element.
    r : int
        Dimension of the reduced basis.
    groupA : ndarray
        Indices of elements with all free DOFs.
    groupB : ndarray
        Indices of elements with some Dirichlet DOFs.
    chunk_size : int
        Number of elements per chunk in groupA.
    n_full_chunks : int
        Number of full-sized chunks in groupA.
    remainder : int
        Number of leftover elements in groupA.
    """

    def __init__(self, form, ubasis: Basis, lob, free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None, nthreads=0, dtype=np.float64):
        """
        Initialize the reduced-order linear form.

        Parameters
        ----------
        form : callable
            Function defining the linear form.
        ubasis : Basis
            Full-order finite element basis for test functions.
        lob : ndarray, shape (N_free, r) or (N, r)
            Reduced basis for load vectors.
        free_dofs : ndarray, optional
            Global indices of free (non-Dirichlet) DOFs.
        mean : ndarray, optional
            Mean snapshot vector subtracted before basis computation.
        nthreads : int, optional
            Number of threads for parallel computation.
        dtype : data-type, optional
            Numeric type for computations.
        """

        super().__init__(form)


        self.r_basis = lob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype
        self.ubasis = ubasis


        # Mapping from full DOFs to indices in the reduced basis.
        self.mapping = self._get_mapping(ubasis)
        self.element_dofs = ubasis.element_dofs
        self.free_indices = self.mapping[self.element_dofs]
        self.mask = self.free_indices >= 0


        self.r = lob.shape[1]
        nonzero_elements = range(ubasis.nelems)


        all_free = np.all(self.mask, axis=0)
        full_free = np.nonzero(all_free)[0]


        self.groupA = np.intersect1d(nonzero_elements, full_free)
        self.groupB = np.setdiff1d(nonzero_elements, self.groupA)


        nA = self.groupA.size
        self.chunk_size=int(nA/2)


        if nA < self.chunk_size:
            self.n_full_chunks = 1
            self.remainder = 0
            self.chunk_size = nA

        else:
            self.n_full_chunks = nA // self.chunk_size
            self.remainder = nA % self.chunk_size


    def _get_mapping(self, basis: Basis) -> ndarray:
        """
        Create mapping from full DOFs to reduced DOF indices.

        Parameters
        ----------
        basis : Basis
            Full-order finite element basis.

        Returns
        -------
        mapping : ndarray, shape (N_full,)
            Array where mapping[i] gives the reduced index of global DOF i, or -1
            for Dirichlet DOFs if free_dofs were provided; otherwise the identity mapping.
        """
        N_full = basis.N
        if self.free_dofs is None:
            return np.arange(N_full)
        else:
            mapping = -np.ones(N_full, dtype=int)
            mapping[self.free_dofs] = np.arange(len(self.free_dofs))
            return mapping
        

    def assemble(self, **kwargs) -> ndarray:
        """
        Assemble the global reduced load vector.

        Projects element load vectors onto reduced bases and sums contributions
        over free DOFs only.

        Parameters
        ----------
        **kwargs
            Additional parameters passed to the form during assembly.

        Returns
        -------
        f_reduced : ndarray, shape (r,)
            Reduced load vector.
        """

        element_vectors = self.extract_element_vector(self.ubasis, **kwargs)

        # Reduced dimension.
        r = self.r

        f_reduced = np.zeros((r,), dtype=self.dtype)


        def compute_reduced_vector_in_chunks(element_vectors):


            mask = self.mask        # shape (Nbfun, n_elements)

            f_reduced_A = np.zeros((r,), dtype=self.dtype)

            for i in range(self.n_full_chunks):

                chunk_idx = self.groupA[i *  self.chunk_size : (i + 1) *  self.chunk_size]
                active_dofs =  self.element_dofs[:, chunk_idx].T  # shape (chunk_size, Nbfun)
                R_chunk = self.r_basis[active_dofs]
                f_chunk = element_vectors[chunk_idx, :]


                f_reduced_A += np.einsum('eia,ei->a', R_chunk, f_chunk, optimize=True)


            if self.remainder > 0:

                chunk_idx = self.groupA[self.n_full_chunks * self.chunk_size :]
                active_dofs = self.element_dofs[:, chunk_idx].T
                R_chunk = self.r_basis[active_dofs] 
                f_chunk = element_vectors[chunk_idx, :]

                f_reduced_A += np.einsum('eia,ei->a', R_chunk, f_chunk, optimize=True)

            def compute_f_red(e):

                local_mask = mask[:, e]

                idx = self.mapping[self.element_dofs[:, e]][local_mask]

                if idx.size == 0:
                    return np.zeros((r,), dtype=self.dtype)
                
                R_free = self.r_basis[idx, :]
                f_local = element_vectors[e, :]
                f_local_free = f_local[local_mask]

                return R_free.T @ (f_local_free)
            
            if self.groupB.size > 0:

                f_func = np.frompyfunc(compute_f_red, 1, 1)
                f_red_B_obj = f_func(self.groupB)
                f_red_B_list = [np.asarray(x, dtype=self.dtype) for x in f_red_B_obj]
                f_reduced_B = np.sum(np.stack(f_red_B_list), axis=0)


            else:

                f_reduced_B = np.zeros((r,), dtype=self.dtype)

            return f_reduced_A + f_reduced_B

        f_reduced = compute_reduced_vector_in_chunks(element_vectors)


        return f_reduced


    def hyperreduction(self, **kwargs) -> ndarray:
        """
        Perform hyperreduction to assemble per-element reduced load contributions.

        Parameters
        ----------
        **kwargs
            Additional parameters passed to the form during hyperreduction.

        Returns
        -------
        f_reduced : ndarray, shape (n_contribs, r)
            Concatenated reduced load contributions for hyperreduction.
        """
        # Extract local element load vectors.
        # element_vectors has shape (n_elements, Nbfun)
        

        element_vectors = self.extract_element_vector(self.ubasis, **kwargs)

        # Reduced dimension.
        r = self.r


        f_reduced = np.empty((0, r), dtype=self.r_basis.dtype)


        for i in range(self.n_full_chunks):

            chunk_idx = self.groupA[i * self.chunk_size : (i + 1) * self.chunk_size]
            active_dofs = self.element_dofs[:, chunk_idx].T  # shape (chunk_size, Nbfun)
            R_chunk = self.r_basis[active_dofs]
            f_chunk = element_vectors[chunk_idx, :]      # shape (chunk_size, Nbfun)


            # Perform a batch-wise operation

            R_chunk_T = np.transpose(R_chunk, (0, 2, 1))             # shape: (chunk_size, r, Nbfun)
            f_chunk_expanded = f_chunk[..., np.newaxis]              # shape: (chunk_size, Nbfun, 1)
            reduced_chunk = np.matmul(R_chunk_T, f_chunk_expanded)     # shape: (chunk_size, r, 1)
            reduced_chunk = reduced_chunk.squeeze(-1)                # shape: (chunk_size, r)
            

            # f_reduced = np.concatenate((f_reduced, np.matmul(R_chunk.T,f_chunk)), axis=0)
            f_reduced = np.concatenate((f_reduced, reduced_chunk), axis=0)



        if self.remainder > 0:

            chunk_idx = self.groupA[self.n_full_chunks * self.chunk_size :]
            active_dofs = self.element_dofs[:, chunk_idx].T
            R_chunk = self.r_basis[active_dofs]#.reshape(chunk_idx.size, element_dofs.shape[0], r)
            f_chunk = element_vectors[chunk_idx, :]


            # Perform a batch-wise operation

            R_chunk_T = np.transpose(R_chunk, (0, 2, 1))             # shape: (chunk_size, r, Nbfun)
            f_chunk_expanded = f_chunk[..., np.newaxis]              # shape: (chunk_size, Nbfun, 1)
            reduced_chunk = np.matmul(R_chunk_T, f_chunk_expanded)     # shape: (chunk_size, r, 1)
            reduced_chunk = reduced_chunk.squeeze(-1)                # shape: (chunk_size, r)


            # f_reduced = np.concatenate((f_reduced, np.matmul(R_chunk.T,f_chunk)), axis=0)
            f_reduced = np.concatenate((f_reduced, reduced_chunk), axis=0)



        def compute_f_red(e):

            local_mask = self.mask[:, e]
            idx = self.mapping[self.element_dofs[:, e]][local_mask]

            if idx.size == 0:
                return np.zeros((r,), dtype=self.dtype)
            
            R_free = self.r_basis[idx, :]
            f_local = element_vectors[e, :]
            f_local_free = f_local[local_mask]
            return R_free.T @ (f_local_free)
        
        if self.groupB.size > 0:

            f_func = np.frompyfunc(compute_f_red, 1, 1)
            f_red_B_obj = f_func(self.groupB)
            f_red_B_list = [np.asarray(x, dtype=self.dtype) for x in f_red_B_obj]
            f_reduced_B = np.stack(f_red_B_list,axis=0)

            f_reduced = np.concatenate((f_reduced, f_reduced_B), axis=0)

        else:

            zero_row = np.zeros((1, r), dtype=self.dtype)
            f_reduced = np.concatenate((f_reduced, zero_row), axis=0)


        del R_chunk, f_chunk, R_chunk_T, f_chunk_expanded, reduced_chunk, f_reduced_B     
        gc.collect()

        return f_reduced
    

    def extract_element_vector(self, basis: AbstractBasis, **kwargs):
        """
        Extract local element load vectors for a linear form.

        Computes per-element load contributions for each local basis function.

        Parameters
        ----------
        basis : AbstractBasis
            Finite element basis associated with the test function.
        **kwargs
            Additional keyword arguments passed to the form.

        Returns
        -------
        element_vectors : ndarray, shape (n_elements, Nbfun)
            Local load vectors for each element, where Nbfun is the number of local basis functions.
        """
        nt = basis.nelems       # Number of elements
        dx = basis.dx           # Quadrature weights per element

        wdict = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        # Allocate an array to hold the local contributions.
        # Shape is (Nbfun, n_elements)
        local_data = np.zeros((basis.Nbfun, nt), dtype=self.dtype)

        if self.nthreads <= 0:
            for i in range(basis.Nbfun):
                local_data[i, :] = self._kernel(basis.basis[i], wdict, dx)
        else:
            # Threaded computation for the linear form.
            def threaded_kernel_vector(data, indices, basis_list, wdict, dx):
                for i in indices:
                    data[i, :] = self._kernel(basis_list[i], wdict, dx)

            indices = np.arange(basis.Nbfun)
            indices_chunks = np.array_split(indices, self.nthreads)
            threads = [
                Thread(
                    target=threaded_kernel_vector,
                    args=(local_data, chunk, basis.basis, wdict, dx)
                )
                for chunk in indices_chunks
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Rearranging data from shape (Nbfun, n_elements) to (n_elements, Nbfun)
        element_vectors = local_data.transpose(1, 0)
        return element_vectors

 

    def _kernel_qp(self, v, w, dx) -> ndarray:
        """
        Return per-element, per-Gauss local vector contributions.

        The returned values include multiplication by ``dx`` but are not yet
        summed over Gauss points.
        """
        val = np.asarray(self.form(*v, w) * dx, dtype=self.dtype)
        if val.ndim != 2:
            raise ValueError(
                "ECM extraction expects the linear form kernel to return a 2D array "
                "of shape (n_elements, n_gauss_points) after multiplying by dx."
            )
        if val.shape == dx.shape:
            return val
        if val.shape == dx.T.shape:
            return val.T
        raise ValueError(
            f"Unexpected ECM linear kernel shape {val.shape}; expected {dx.shape} or {dx.T.shape}."
        )

    def _threaded_kernel_qp(self, data, indices, basis_list, wdict, dx):
        for i in indices:
            data[i, :, :] = self._kernel_qp(basis_list[i], wdict, dx)

    def extract_element_vector_qp(self, basis: AbstractBasis, **kwargs):
        """
        Extract local element load vectors resolved at Gauss points.

        Returns
        -------
        element_vectors_q : ndarray, shape (n_elements, n_local_dofs, n_q)
            Local load contributions already multiplied by ``dx`` but not yet
            summed over quadrature points.
        """
        nt = basis.nelems
        dx = basis.dx
        nq = dx.shape[1]

        wdict = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        local_data = np.zeros((basis.Nbfun, nt, nq), dtype=self.dtype)

        if self.nthreads <= 0:
            for i in range(basis.Nbfun):
                local_data[i, :, :] = self._kernel_qp(basis.basis[i], wdict, dx)
        else:
            indices = np.arange(basis.Nbfun)
            indices_chunks = np.array_split(indices, self.nthreads)
            threads = [
                Thread(
                    target=self._threaded_kernel_qp,
                    args=(local_data, chunk, basis.basis, wdict, dx),
                )
                for chunk in indices_chunks
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        return np.transpose(local_data, (1, 0, 2))

    def hyperreduction_ecm(self, **kwargs) -> ndarray:
        """
        Assemble per-element, per-Gauss reduced residual contributions for ECM.

        Returns
        -------
        f_reduced_q : ndarray, shape (n_elements * n_q, r)
            Reduced contributions ordered with flat index ``e * n_q + q``.
        """
        element_vectors_q = self.extract_element_vector_qp(self.ubasis, **kwargs)
        n_elements, _, n_q = element_vectors_q.shape
        r = self.r

        rows = []
        for e in range(n_elements):
            local_mask = self.mask[:, e]
            idx = self.mapping[self.element_dofs[:, e]][local_mask]
            if idx.size == 0:
                rows.append(np.zeros((n_q, r), dtype=self.dtype))
                continue

            R_free = self.r_basis[idx, :]
            f_local_free_q = element_vectors_q[e, local_mask, :]
            f_red_q = (R_free.T @ f_local_free_q).T
            rows.append(np.asarray(f_red_q, dtype=self.dtype))

        return np.vstack(rows)





