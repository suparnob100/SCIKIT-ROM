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
  
  [Author: Suparno Bhattacharyya]
"""
from typing import Optional
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.basis import AbstractBasis
from skfem.assembly.form.bilinear_form import BilinearForm  
from numpy.typing import DTypeLike 

class BilinearFormHYPERROM_ecsw(BilinearForm):
    """
    Reduced-order bilinear form for hyper-reduction of stiffness matrices.

    Projects element-level stiffness matrices onto reduced bases and
    assembles the global reduced stiffness matrix. Handles Dirichlet
    boundary conditions via mapping from full degrees of freedom (DOFs)
    to free (non-Dirichlet) DOFs. All reduced bases and operations
    are performed only on free DOFs, with Dirichlet and mean field
    contributions reinserted during solution reconstruction.

    Parameters
    ----------
    form : callable
        The original bilinear form function taking test and trial basis
        functions and assembly parameters.
    elem_weight : scalar or ndarray
        Element-wise weights (e.g., quadrature or sampling weights). Can be
        a single scalar or an array of length equal to the number of elements.
    ubasis : Basis
        Trial-space reduced basis object containing full DOF count and
        element connectivity data.
    lob : ndarray
        Left (test) reduced basis matrix of shape (N_free, r) if
        free_dofs is provided, or (N, r) otherwise.
    rob : ndarray
        Right (trial) reduced basis matrix, with same shape requirements as
        `lob`.
    vbasis : Basis, optional
        Reduced basis for test functions; if None, defaults to `ubasis`.
    free_dofs : ndarray of int, optional
        Indices of global DOFs that are free (non-Dirichlet). If provided,
        bases are defined only on these DOFs.
    mean : ndarray, optional
        Mean snapshot vector of length N_full DOFs, subtracted during basis
        computation and reinserted during reconstruction.
    nthreads : int, optional
        Number of threads for element-wise assembly operations. Default 0
        (serial execution).
    dtype : data-type, optional
        NumPy data type for assembled matrices and intermediate arrays.

    Attributes
    ----------
    lob : ndarray
        Left reduced basis (possibly restricted to free DOFs).
    rob : ndarray
        Right reduced basis (possibly restricted to free DOFs).
    free_dofs : ndarray or None
        Indices of free DOFs if Dirichlet conditions are present.
    mean : ndarray or None
        Mean snapshot vector for solution centering.
    r : int
        Reduced dimension (number of basis vectors).
    mapping : ndarray of int
        Mapping from full DOF indices to reduced free-DOF indices.
    cluster_idx : list of ndarray
        Indices of elements grouped by number of free DOFs per element.
    order_cluster : list of ndarray
        Local ordering for extracting free DOF positions within each cluster.
    w_cluster : list of ndarray
        Element weights corresponding to each cluster.
    R_test_free : list of ndarray
        Test-basis rows restricted to free DOFs per element cluster.
    R_trial_free : list of ndarray
        Trial-basis rows restricted to free DOFs per element cluster.

    Notes
    -----
    - Clustering by element free DOF count enables vectorized extraction
      of submatrices for each element group, reducing Python looping.
    - Uses Einstein summation (`np.einsum`) to contract element-level
      contributions into the reduced global stiffness matrix.
    """

    def __init__(self, form, elem_weight, ubasis: Basis, lob, rob,
                 vbasis: Optional[Basis] = None, free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None, nthreads: int = 0,
                 dtype: DTypeLike = np.float64):
        """
        Initialize the reduced-order bilinear form and preprocess element clusters.

        Parameters
        ----------
        form : callable
            The original bilinear form function to assemble local matrices.
        elem_weight : scalar or ndarray
            Weight(s) applied to each element during assembly.
        ubasis : Basis
            Trial-space reduced basis with full DOF count and element mapping.
        lob : ndarray
            Left reduced basis (test functions), shape matching `rob`.
        rob : ndarray
            Right reduced basis (trial functions), shape matching `lob`.
        vbasis : Basis, optional
            Test-space basis; defaults to `ubasis` if None.
        free_dofs : ndarray of int, optional
            Indices of non-Dirichlet DOFs. If None, all DOFs are free.
        mean : ndarray, optional
            Mean vector removed from snapshots during basis computation.
        nthreads : int, default 0
            Number of threads for parallel assembly (_get_mapping excluded).
        dtype : data-type, default np.float64
            Data type for internal arrays and computed matrices.

        Raises
        ------
        ValueError
            If provided `free_dofs` length does not match basis DOF count.
        """

        super().__init__(form)

        # ---------------- core variables ----------------
        self.lob = lob
        self.rob = rob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype
        self.ubasis = ubasis
        
        if vbasis is None:
            self.vbasis = ubasis

        self.r = lob.shape[1]

        # ------------- weights / sampled mesh --------
        self.weight = np.array(elem_weight)
        self.nonzero_elements = np.nonzero(elem_weight)[0]
        self.ubasis_rom = ubasis.with_elements(self.nonzero_elements)
        if vbasis is None:
            self.vbasis_rom = self.ubasis_rom
        else:
            self.vbasis_rom = vbasis.with_elements(self.nonzero_elements)

        # ------------- mapping & masks ------------------
        self.mapping = self._get_mapping(self.ubasis_rom)
        self.element_dofs = self.ubasis_rom.element_dofs
        self.free_indices = self.mapping[self.element_dofs]
        self.mask = self.free_indices >= 0
        self.n_freedom = np.sum(self.mask, axis=0)
        self.unique_freedom = np.unique(self.n_freedom)

        # --------- per-cluster caches --------
        self.cluster_idx=[]
        self.order_cluster=[]
        self.w_cluster=[]
        self.R_test_free=[]
        self.R_trial_free=[]

        # print(self.unique_freedom)

        # Loop over clusters (the number of clusters is small, one per unique free DOF count).
        for nf in self.unique_freedom:


            # Get indices (into the restricted arrays) of elements that have exactly nf free DOFs.
            self.cluster_idx.append(np.nonzero(self.n_freedom == nf)[0])
            # cluster_size = self.cluster_idx[-1].size


            # --- Vectorized extraction within the cluster ---
            # Restrict the free_mask and free_indices arrays to this cluster.
            # fm_cluster: shape (n_local_dofs, cluster_size)
            fm_cluster = self.mask[:, self.cluster_idx[-1]]
            fi_cluster = self.free_indices[:, self.cluster_idx[-1]]


            # For each column in fm_cluster (i.e. each element), we wish to extract the indices 
            # (in the local DOF space) where the mask is True.
            # Since each column has exactly nf True entries, we can use argsort along axis=0.
            # Booleans are sorted as False (0) then True (1); thus, the last nf indices in each column
            # correspond to free DOF positions.
            order = np.argsort(fm_cluster, axis=0)[-nf:, :]   # shape: (nf, cluster_size)
            self.order_cluster.append(order.T)                           # shape: (cluster_size, nf))


            # Using advanced indexing, extract the free DOF indices for each element.
            # Broadcast column indices to match order's shape.
            cluster_size = order.shape[1]
            col_idx = np.tile(np.arange(cluster_size), (nf, 1))       # shape: (nf, cluster_size)


            # free_dofs_cluster: shape (cluster_size, nf); each row contains the local indices 
            # (in the element's DOF array) of the free DOFs.
            # free_dofs_cluster = fi_cluster[order, col_idx].T
            free_dofs_cluster = np.take_along_axis(fi_cluster, order, axis=0).T.astype(int)



            # Now, extract the rows from the projection matrices.
            # self.lob and self.rob are assumed to be arrays of shape (global_dofs, r).
            # Using free_dofs_cluster as indices produces:
            # R_test_free, R_trial_free: shape (cluster_size, nf, r)


            self.R_test_free.append(self.lob[free_dofs_cluster])
            self.R_trial_free.append(self.rob[free_dofs_cluster])


            # Gather the weights for the current cluster.
            if np.ndim(self.weight) == 0:
                self.w_cluster.append(self.weight * np.ones(cluster_size, dtype=self.dtype))

            else:
                # weight for restricted elements is weight[nonzero_elements]; then index by cluster_idx.
                self.w_cluster.append(self.weight[self.nonzero_elements][self.cluster_idx[-1]])


    def _get_mapping(self, basis: Basis) -> ndarray:
        """
        Build a mapping from global DOFs to reduced basis indices.

        If `free_dofs` was provided, Dirichlet DOFs map to -1 and free DOFs map
        to [0, N_free-1]. Otherwise, returns the identity mapping.

        Parameters
        ----------
        basis : Basis
            Basis object containing total DOF count and `free_dofs` attribute.

        Returns
        -------
        mapping : ndarray of int
            Array of length N_full where mapping[i] gives the reduced-basis index
            of global DOF i, or -1 for Dirichlet DOFs.
        """
        N_full = basis.N
        if self.free_dofs is None:
            return np.arange(N_full)
        else:
            mapping = np.arange(basis.N)
            non_free_dofs = np.setdiff1d(mapping, self.free_dofs)
            mapping[non_free_dofs]=-1
            return mapping



    def assemble_weighted_ecsw(self, **kwargs):

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

        
        element_matrices = self.extract_element_matrices_rom(self.ubasis_rom, self.vbasis_rom, elem_indices=self.nonzero_elements, **kwargs)

        # Now, n_elements in the restricted basis equals len(nonzero_elements)
        K_reduced = np.zeros((self.r, self.r), dtype=self.dtype)

        i = 0        


        # Loop over clusters (the number of clusters is small, one per unique free DOF count).
        for nf in self.unique_freedom:


            # Next, extract the free submatrices from the local element matrices.
            # For the current cluster, restrict element_matrices.
            # elem_mat_cluster: shape (cluster_size, n_local_dofs, n_local_dofs)
            elem_mat_cluster = element_matrices[self.cluster_idx[i], :, :]


            # For each element, we need rows and columns corresponding to the free DOF positions.
            # Use np.take_along_axis twice (once for rows, then for columns).
            # First, select free rows.
            K_rows = np.take_along_axis(elem_mat_cluster, self.order_cluster[i][:, :, None], axis=1) 


            # Then, select free columns from the result.
            # K_cluster_free: shape (cluster_size, nf, nf)
            K_cluster_free = np.take_along_axis(K_rows, self.order_cluster[i][:, None, :], axis=2)


            # --- Vectorized contraction over the cluster ---
            # Each element e in the cluster contributes:
            #   A_e = (R_test_free[e])^T @ (w_cluster[e]*K_cluster_free[e]) @ R_trial_free[e]
            # We sum these contributions in one vectorized operation using np.einsum.
            # Breakdown of indices:
            #   e: element in cluster (0 <= e < cluster_size)
            #   a,b: local free DOF indices (0 <= a,b < nf)
            #   i,j: reduced DOF indices (0 <= i,j < r)
            cluster_contribution = np.einsum(
                'ein,enm,emj->ij',
                self.R_test_free[i].transpose(0, 2, 1),               # shape: (cluster_size, r, nf)
                K_cluster_free * self.w_cluster[i][:, None, None],     # shape: (cluster_size, nf, nf)
                self.R_trial_free[i],                                   # shape: (cluster_size, nf, r)
                optimize=True                                   
            )
            K_reduced += cluster_contribution

            i+=1

        return K_reduced



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