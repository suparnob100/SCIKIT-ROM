"""
ECSW-based hyperreduction for finite element bilinear forms with element clustering.

This module provides hyperreduction of bilinear forms using Energy-Conserving 
Sampling and Weighting (ECSW) combined with intelligent element clustering for 
efficient reduced-order stiffness assembly. It achieves dramatic computational 
speedups by clustering elements by number of free DOFs for vectorized operations,
extracting and projecting element stiffness blocks onto reduced bases with weights,
and assembling global reduced matrices via vectorized Einstein summation.

TL;DR
-----
Enables substantial speedup in bilinear form assembly for ROMs while 
preserving stability and energy conservation through intelligent element clustering
and weighted assembly strategies.

Author
------
Suparno Bhattacharyya
"""


from typing import Optional, Any
from types import MethodType
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.basis import AbstractBasis, FacetBasis
from skfem.assembly.form.bilinear_form import BilinearForm  
from numpy.typing import DTypeLike 



def with_elements(self, elements: Optional[Any] = None) -> 'FacetBasis':
    """
    Return a similar basis on a subset of element indices.
    
    **TL;DR**: Creates a restricted FacetBasis containing only specified elements.
    
    Parameters
    ----------
    elements : Optional[Any], optional
        Subset of element indices to restrict the basis to
    
    Returns
    -------
    FacetBasis
        New FacetBasis instance restricted to specified elements
    """
    return type(self)(
        self.mesh,
        self.elem,
        mapping=self.mapping,
        quadrature=self.quadrature,
        facets=elements,
    )



class BilinearFormHYPERROM_ecsw(BilinearForm):
    """
    ECSW-based hyperreduced bilinear form with element clustering for efficient assembly.

    **TL;DR**: Dramatically accelerates bilinear form assembly substantially through 
    energy-conserving element clustering and weighted sampling, providing both 
    computational efficiency and numerical stability for real-time ROM applications.

    This class implements a sophisticated hyperreduction strategy that combines 
    Energy-Conserving Sampling and Weighting (ECSW) with intelligent element 
    clustering to achieve massive computational savings while preserving crucial 
    physical properties. The approach works through several key innovations:
    
    - Element clustering by free DOF count for vectorized operations
    - Energy-conserving weighted assembly preserving physical properties
    - Efficient submatrix extraction using advanced NumPy indexing
    - Vectorized Einstein summation for parallel element contributions
    
    The hyperreduction is particularly effective for problems where:
    
    - Energy conservation is critical (structural dynamics, wave propagation)
    - Element distributions are relatively uniform (similar local DOF counts)
    - Computational stability is paramount for long-time integration
    - Real-time performance is required for control or optimization

    Parameters
    ----------
    form : callable
        The original bilinear form function to be hyperreduced. Should accept 
        test and trial basis functions and return element-wise stiffness contributions
    elem_weight : scalar or array_like of shape (n_elements,)
        Element-wise ECSW weights determining the contribution of each element 
        to the reduced assembly. Can be a single scalar applied to all elements 
        or individual weights per element from ECSW analysis
    ubasis : Basis
        Trial-space finite element basis containing full DOF count, element 
        connectivity, and quadrature information for the original mesh
    lob : ndarray of shape (n_free, r) or (n_full, r)
        Left (test) reduced basis matrix. Shape depends on whether free_dofs 
        is provided - if so, basis is defined only on free DOFs
    rob : ndarray of shape (n_free, r) or (n_full, r)
        Right (trial) reduced basis matrix with same shape requirements as lob.
        Projects full-order solutions to the r-dimensional reduced space
    vbasis : Basis, optional
        Test-space finite element basis. If None, defaults to ubasis for 
        standard Galerkin formulations
    free_dofs : ndarray of int, optional
        Indices of global DOFs that are free (non-Dirichlet). If provided,
        all reduced bases and operations are performed only on these DOFs
    mean : ndarray, optional
        Mean snapshot vector for solution centering. Required if snapshot data 
        was mean-subtracted during reduced basis construction
    nthreads : int, default=0
        Number of threads for parallel element matrix extraction. Zero means 
        serial execution, positive values enable multi-threaded assembly
    dtype : numpy.dtype, default=np.float64
        Numerical precision for all computations and storage arrays

    Attributes
    ----------
    lob : ndarray
        Left reduced basis matrix, possibly restricted to free DOFs
    rob : ndarray
        Right reduced basis matrix, possibly restricted to free DOFs
    free_dofs : ndarray or None
        Indices of free DOFs if Dirichlet boundary conditions are present
    mean : ndarray or None
        Mean snapshot vector for solution centering and reconstruction
    r : int
        Reduced dimension (number of reduced basis vectors)
    mapping : ndarray of int
        Mapping from full DOF indices to reduced free-DOF indices, with 
        Dirichlet DOFs mapped to -1
    cluster_idx : list of ndarray
        Element indices grouped by number of free DOFs per element. Each entry 
        contains indices of elements with the same free DOF count
    order_cluster : list of ndarray
        Local DOF ordering within each cluster for efficient submatrix extraction.
        Shape: (cluster_size, n_free_dofs_in_cluster)
    w_cluster : list of ndarray
        ECSW weights corresponding to elements in each cluster
    R_test_free : list of ndarray
        Test basis matrices restricted to free DOFs for each cluster.
        Shape: (cluster_size, n_free_dofs, r)
    R_trial_free : list of ndarray
        Trial basis matrices restricted to free DOFs for each cluster.
        Shape: (cluster_size, n_free_dofs, r)
    unique_freedom : ndarray of int
        Unique counts of free DOFs per element, determining the number of clusters
    weight : ndarray
        Array of ECSW weights
    nonzero_elements : ndarray
        Indices of elements with non-zero weights
    ubasis_rom : Basis
        Trial basis restricted to non-zero weight elements
    vbasis_rom : Basis
        Test basis restricted to non-zero weight elements
    element_dofs : ndarray
        Element-to-DOF connectivity for restricted mesh
    free_indices : ndarray
        Free DOF indices for each element
    mask : ndarray of bool
        Boolean mask indicating which DOFs are free in each element
    n_freedom : ndarray of int
        Number of free DOFs per element
    """


    def __init__(self, form, elem_weight, ubasis: Basis, lob, rob,
                 vbasis: Optional[Basis] = None, free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None, nthreads: int = 0,
                 dtype: DTypeLike = np.float64):
        """
        Initialize hyperreduced bilinear form with ECSW weights and element clustering.

        **TL;DR**: Sets up element clustering, DOF mapping, and reduced basis projections
        for efficient hyperreduced assembly.

        Parameters
        ----------
        form : callable
            The original bilinear form function to assemble local matrices
        elem_weight : scalar or ndarray
            Weight(s) applied to each element during assembly from ECSW analysis
        ubasis : Basis
            Trial-space reduced basis with full DOF count and element mapping
        lob : ndarray
            Left reduced basis (test functions), shape matching `rob`
        rob : ndarray
            Right reduced basis (trial functions), shape matching `lob`
        vbasis : Basis, optional
            Test-space basis; defaults to `ubasis` if None
        free_dofs : ndarray of int, optional
            Indices of non-Dirichlet DOFs. If None, all DOFs are treated as free
        mean : ndarray, optional
            Mean vector removed from snapshots during basis computation
        nthreads : int, default=0
            Number of threads for parallel assembly operations
        dtype : data-type, default=np.float64
            Data type for internal arrays and computed matrices
        """
        super().__init__(form)

        # ---------------- core variables ----------------
        self.lob = lob
        self.rob = rob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype

        if isinstance(ubasis, FacetBasis) and not hasattr(ubasis, "with_elements"):
            ubasis.with_elements = MethodType(with_elements, ubasis)

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

        # Loop over clusters (the number of clusters is small, one per unique free DOF count).
        for nf in self.unique_freedom:

            # Get indices (into the restricted arrays) of elements that have exactly nf free DOFs.
            self.cluster_idx.append(np.nonzero(self.n_freedom == nf)[0])

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
        Build mapping from global DOFs to reduced basis indices for Dirichlet handling.

        **TL;DR**: Creates efficient integer mapping that transforms global DOF 
        indices to free DOF indices, enabling seamless boundary condition treatment 
        in the reduced-order framework.

        This method constructs a crucial data structure that enables efficient 
        handling of Dirichlet boundary conditions in the reduced-order context. 
        The mapping allows the algorithm to:
        
        - Identify which global DOFs are free (non-Dirichlet)
        - Map free DOFs to contiguous reduced basis indices [0, N_free-1]
        - Mark Dirichlet DOFs with sentinel value -1 for easy identification
        - Enable vectorized operations on free DOF subsets

        Parameters
        ----------
        basis : Basis
            Finite element basis object containing total DOF count and connectivity.
            The basis.N attribute provides the total number of global DOFs

        Returns
        -------
        mapping : ndarray of int, shape (N_full,)
            Integer array where mapping[i] gives the reduced-basis index of global 
            DOF i, or -1 if DOF i is a Dirichlet (constrained) DOF
            
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
        Assemble the globally weighted reduced stiffness matrix using ECSW.

        **TL;DR**: Main assembly method that orchestrates element clustering, 
        vectorized stiffness extraction, ECSW weighting, and reduced basis 
        projection to produce the final r×r reduced-order stiffness matrix.

        This method performs the complete ECSW hyperreduction assembly process 
        through a sophisticated multi-stage algorithm:

        1. **Element Matrix Extraction**: Calls element extraction routines to 
           compute local stiffness matrices for all active elements, leveraging 
           parallel processing when available.

        2. **Cluster-Based Processing**: Processes elements in clusters based on 
           their free DOF count, enabling highly efficient vectorized operations 
           and eliminating expensive Python loops.

        3. **Submatrix Extraction**: For each cluster, extracts the free DOF 
           submatrices from local element matrices using advanced NumPy indexing 
           for maximum efficiency.

        4. **ECSW Weighting**: Applies energy-conserving weights to preserve 
           physical properties while enabling computational reduction.

        5. **Vectorized Contraction**: Uses Einstein summation to perform 
           parallel contractions over entire clusters: 
           A_reduced += Σ_e R_test[e]^T @ (w[e] * K_local[e]) @ R_trial[e]

        The final result preserves the mathematical structure of the full-order 
        operator while achieving dramatic computational savings through intelligent 
        clustering and vectorization strategies.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to element extraction routines
            for controlling assembly behavior, such as material parameters or 
            quadrature settings

        Returns
        -------
        K_reduced : ndarray of shape (r, r)
            Assembled reduced-order stiffness matrix ready for use in ROM systems.
            This matrix preserves the energy conservation properties of the 
            full-order operator while enabling real-time evaluation
        
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
        Extract element stiffness matrices for hyperreduced mesh assembly.

        **TL;DR**: Computes local element stiffness matrices for the reduced 
        element set using either serial or parallel execution, providing the 
        fundamental building blocks for ECSW-weighted global assembly.

        This method performs the core finite element integration to compute 
        element-level contributions to the global bilinear form. The integration 
        is performed only over elements selected by the hyperreduction strategy, 
        dramatically reducing computational cost while maintaining accuracy through 
        ECSW weighting.

        The method supports both execution modes:
        
        - **Serial Mode** (nthreads=0): Sequential element-by-element computation
        - **Parallel Mode** (nthreads>0): Multi-threaded parallel element processing

        For each element, the method evaluates the bilinear form:
        K_e[i,j] = ∫_Ω_e φ_i(x) * form * φ_j(x) dx

        where φ_i, φ_j are basis functions and integration uses the quadrature 
        rules embedded in the finite element basis.

        Parameters
        ----------
        ubasis : Basis
            Trial-space finite element basis containing mesh connectivity,
            quadrature points, and basis function evaluations
        vbasis : Basis, optional
            Test-space finite element basis. If None, defaults to ubasis 
            for standard Galerkin formulations
        elem_indices : array_like of int, optional
            Specific element indices to include in extraction. If None,
            processes all elements in the hyperreduced mesh
        **kwargs : dict
            Additional keyword arguments passed to the bilinear form evaluation,
            such as material parameters or other problem-specific data

        Returns
        -------
        element_matrices : ndarray of shape (n_elements, n_local_dofs, n_local_dofs)
            Array of local element stiffness matrices. Each element_matrices[e] 
            contains the n_local_dofs × n_local_dofs stiffness matrix for element e

        Raises
        ------
        ValueError
            If trial and test bases have incompatible quadrature point counts,
            indicating a mismatch in integration rules
            
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
