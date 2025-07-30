"""
Implements reduced-order bilinear form assembly for full-order to reduced-order transformations.

This module provides:
  - `BilinearFormROM`: a subclass of `skfem.assembly.form.bilinear_form.BilinearForm`
    that projects full-order element stiffness matrices onto reduced bases,
    groups elements by Dirichlet-free and mixed-Dirichlet sets for memory-efficient handling,
    and assembles the global reduced stiffness matrix with optional chunked computation.

The `rom` folder contains core tools for reduced-order modeling (ROM), including:
  - Classes for projecting and assembling reduced-order bilinear and linear forms
  - Utilities for handling Dirichlet boundary conditions in reduced spaces
  - Chunked and clustered assembly routines to manage large-scale stiffness/load data
  - Mapping utilities between full-order and reduced-order degrees of freedom
"""
from typing import Optional
from threading import Thread


import numpy as np
from numpy import ndarray

from skfem.assembly.basis import Basis
from skfem.assembly.form.coo_data import COOData
from skfem.assembly.form import Form
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.basis import AbstractBasis
from skfem.assembly.form.bilinear_form import BilinearForm  


class BilinearFormROM(BilinearForm):
    """
    BilinearFormROM

    Bilinear form that projects element stiffness matrices onto reduced bases
    and assembles the global reduced stiffness matrix, handling Dirichlet boundary
    conditions via mappings from full to free DOFs.

    Attributes
    ----------
    lob : ndarray, shape (N_free, r) or (N, r)
        Left (test) reduced basis.
    rob : ndarray, shape (N_free, r) or (N, r)
        Right (trial) reduced basis.
    free_dofs : ndarray or None
        Indices of global free (non-Dirichlet) DOFs.
    mean : ndarray or None
        Mean snapshot vector subtracted before basis computation.
    nthreads : int
        Number of threads for parallel computation.
    dtype : data-type
        Numeric type for computations.
    ubasis : Basis
        Full-order finite element basis for trial functions.
    vbasis : Basis
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

    def __init__(
        self,
        form,
        ubasis: Basis,
        lob,
        rob,
        vbasis: Optional[Basis] = None,
        free_dofs: Optional[ndarray] = None,
        mean: Optional[ndarray] = None,
        nthreads: int = 0,
        dtype: np.dtype = np.float64,
    ):
        """
        Initialize the reduced-order bilinear form.

        Parameters
        ----------
        form : callable
            Function defining the bilinear form.
        ubasis : Basis
            Full-order finite element basis for trial functions.
        lob : ndarray, shape (N_free, r) or (N, r)
            Left (test) reduced basis.
        rob : ndarray, shape (N_free, r) or (N, r)
            Right (trial) reduced basis.
        vbasis : Basis, optional
            Finite element basis for test functions. If None, defaults to ubasis.
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
        self.lob = lob
        self.rob = rob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype
        self.ubasis = ubasis
        
        if vbasis is None:
            self.vbasis = ubasis

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



    def _get_mapping(self, ubasis: Basis) -> ndarray:
        """
        Create mapping from full DOFs to reduced DOF indices.

        Parameters
        ----------
        ubasis : Basis
            Full-order finite element basis.

        Returns
        -------
        mapping : ndarray, shape (N_full,)
            Array where mapping[i] gives the reduced index of global DOF i, or -1
            for Dirichlet DOFs if free_dofs were provided; otherwise the identity mapping.
        """
        N_full = ubasis.N
        if self.free_dofs is None:
            return np.arange(N_full)
        else:
            mapping = -np.ones(N_full, dtype=int)
            mapping[self.free_dofs] = np.arange(len(self.free_dofs))
            return mapping

#### Ideally we do not want to use this assembly method, but it is kept for situations where the full-order stiffness matrix cannot assembled in one go due to memory constraints.

    def assemble(self, vbasis: Optional[Basis] = None, **kwargs):

        """
        Assemble the global reduced stiffness matrix.

        Projects element stiffness matrices onto reduced bases and sums contributions
        over free DOFs only.

        Parameters
        ----------
        vbasis : Basis, optional
            Finite element basis for test functions. Defaults to ubasis.
        **kwargs
            Additional parameters passed to the form during assembly.

        Returns
        -------
        K_reduced : ndarray, shape (r, r)
            Reduced stiffness matrix.
        """
        ubasis = self.ubasis
        vbasis = self.vbasis
        
        # Extract local element stiffness matrices.
        element_matrices = self.extract_element_matrices(ubasis, vbasis, **kwargs)

        # Reduced dimension.
        r = self.r

        # Global reduced stiffness matrix.
        K_reduced = np.zeros((r, r), dtype=self.dtype)


        def compute_reduced_matrix(lob, rob, element_matrices,  dtype=np.float64):

            free_indices = self.free_indices
            mask = self.mask

            sum_A = np.zeros((r, r), dtype=dtype)


            for i in range(self.n_full_chunks):

                chunk_indices = self.groupA[i * self.chunk_size : (i + 1) * self.chunk_size]
                active_dofs_chunk = self.element_dofs.T[chunk_indices]
                R_test_chunk = lob[active_dofs_chunk]
                R_trial_chunk = rob[active_dofs_chunk]
                K_local_chunk = element_matrices[chunk_indices, :, :]

                sum_A += np.einsum('eia,eij,ejb->ab', R_test_chunk, K_local_chunk, R_trial_chunk, optimize=True)


            if self.remainder > 0:

                chunk_indices = self.groupA[self.n_full_chunks * self.chunk_size :]
                active_dofs_chunk = self.element_dofs.T[chunk_indices]
                R_test_chunk = lob[active_dofs_chunk]
                R_trial_chunk = rob[active_dofs_chunk]
                K_local_chunk = element_matrices[chunk_indices, :, :]

                sum_A += np.einsum('eia,eij,ejb->ab', R_test_chunk, K_local_chunk, R_trial_chunk, optimize=True)


            def compute_K_red_for_element(e):

                idx = free_indices[:, e][mask[:, e]]

                if idx.size == 0:

                    return np.zeros((r, r), dtype=dtype)
                
                R_test = lob[idx, :]
                R_trial = rob[idx, :]
                K_loc = element_matrices[e, :, :]
                K_loc_free = K_loc[np.ix_(mask[:, e], mask[:, e])]

                return (R_test.T @ K_loc_free @ R_trial)


            if self.groupB.size > 0:

                f = np.frompyfunc(compute_K_red_for_element, 1, 1)
                K_red_B_obj = f(self.groupB)
                K_red_B_list = [np.asarray(x, dtype=dtype) for x in K_red_B_obj]
                K_red_B = np.sum(np.stack(K_red_B_list), axis=0)


            else:

                K_red_B = np.zeros((r, r), dtype=dtype)


            K_reduced = sum_A + K_red_B

            return K_reduced

        K_reduced = compute_reduced_matrix(self.lob, self.rob, element_matrices)

        return K_reduced



    def extract_element_matrices(self, ubasis: Basis, vbasis=None, **kwargs):
        """
        Extract local element stiffness matrices for a given bilinear form.

        Parameters
        ----------
        form : BilinearForm
            A bilinear form instance (e.g., one decorated with @BilinearForm).
        ubasis : Basis
            The finite element basis associated with the trial function.
        vbasis : Basis, optional
            The finite element basis associated with the test function.
            If None, vbasis is set equal to ubasis.
        kwargs : dict, optional
            Additional keyword arguments to be passed as extra parameters
            during the assembly process.

        Returns
        -------
        element_matrices : ndarray
            A NumPy array of shape (n_elements, Nbfun, Nbfun) containing
            the local stiffness matrices for each element, where Nbfun is the
            number of local basis functions per element.
        """
        if vbasis is None:
            vbasis = ubasis
        elif ubasis.X.shape[-1] != vbasis.X.shape[-1]:
            raise ValueError("Quadrature mismatch: trial and test functions should have the same number of integration points.")

        nt = ubasis.nelems         # Number of elements
        dx = ubasis.dx             # Quadrature weights per element


        # Combine default parameters with any additional keyword arguments.
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })


        # Allocate an array to hold the local contributions.
        # Its shape is (Nbfun, Nbfun, n_elements)
        local_data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=self.dtype)

        # Loop over local basis indices (or use threading if requested)
        if self.nthreads <= 0:
            for j in range(ubasis.Nbfun):
                for i in range(vbasis.Nbfun):
                    local_data[j, i, :] = self._kernel(
                        ubasis.basis[j],
                        vbasis.basis[i],
                        wdict,
                        dx,
                    )
        else:
            # Prepare index pairs for threaded computation.
            indices = np.array([[i, j]
                                for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))])
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

        # Rearrange data from (Nbfun, Nbfun, n_elements) to (n_elements, Nbfun, Nbfun)
        element_matrices = local_data.transpose(2, 0, 1)
        return element_matrices


 