"""ECSW hyper-reduction for linear forms.

TL;DR
-----
This module assembles reduced load vectors with sparse ECSW element weights.

Notes
-----
It clusters active elements, projects local load vectors, and accumulates weighted reduced vectors.
"""

from typing import Optional, Any
from types import MethodType
from threading import Thread
import numpy as np
from numpy import ndarray
from skfem.assembly.basis import Basis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.linear_form import LinearForm  # Import the full-order linear form class
from skfem.assembly.basis import AbstractBasis, FacetBasis
from numpy.typing import DTypeLike 

def with_elements(self, elements: Optional[Any] = None) -> 'FacetBasis':
    """Return a similar basis on a subset of element indices.
    
    TL;DR
    -----
    Return a similar basis on a subset of element indices.
    """
    return type(self)(
        self.mesh,
        self.elem,
        mapping=self.mapping,
        quadrature=self.quadrature,
        facets=elements,
    )

class LinearFormHYPERROM_ecsw(LinearForm):
    """Reduced-order linear form for hyper-reduction of load vectors.
    
    TL;DR
    -----
    Reduced-order linear form for hyper-reduction of load vectors.
    
    Notes
    -----
    Projects element-level load vectors onto a reduced basis and assembles the
    global reduced load vector. Handles Dirichlet boundary conditions via mapping
    from full degrees of freedom (DOFs) to free (non-Dirichlet) DOFs. All operations
    occur only on free DOFs, with Dirichlet and mean field contributions reinserted
    during reconstruction.
    
    Parameters
    ----------
    form : callable
        The original linear form function evaluating local load contributions.
    elem_weight : scalar or ndarray
        Element-wise weights (e.g., quadrature or sampling weights). Can be a
        single scalar or an array of length equal to the number of elements.
    ubasis : Basis
        Finite element basis object with full DOF count and element connectivity.
    lob : ndarray
        Reduced basis matrix of shape (N_free, r) if `free_dofs` is provided,
        or (N, r) otherwise, where r is the reduced dimension.
    free_dofs : ndarray of int, optional
        Indices of global DOFs that are free (non-Dirichlet). If provided, basis
        is defined only on these DOFs.
    mean : ndarray, optional
        Mean snapshot vector of length N_full DOFs, subtracted during basis computation
        and reinserted during reconstruction.
    nthreads : int, optional
        Number of threads for parallel element-wise evaluation. Default is 0 (serial).
    dtype : data-type, optional
        NumPy data type for assembled vectors and intermediate arrays.
    """

    def __init__(self, form, elem_weight, ubasis: Basis, lob, free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None, nthreads=0,dtype: DTypeLike = np.float64):
        """Initialize the reduced-order linear form and preprocess element clusters.
        
        TL;DR
        -----
        Initialize the reduced-order linear form and preprocess element clusters.
        
        Parameters
        ----------
        form : callable
            The original linear form evaluator for load extraction.
        elem_weight : scalar or ndarray
            Weight(s) applied to each element during assembly.
        ubasis : Basis
            Basis defining element connectivity and DOF counts.
        lob : ndarray
            Reduced basis for load projection (shape matches trial basis in bilinear forms).
        free_dofs : ndarray of int, optional
            Indices of DOFs not subject to Dirichlet conditions; if None,
            all DOFs are considered free.
        mean : ndarray, optional
            Mean vector removed from snapshots during basis formation.
        nthreads : int, default 0
            Number of threads for parallel evaluation of element vectors.
        dtype : data-type, default np.float64
            Data type for intermediate and output arrays.
        
        Raises
        ------
        ValueError
            If `free_dofs` length does not match the full DOF count in `ubasis`.
        """

        super().__init__(form)

        self.r_basis = lob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype


        if isinstance(ubasis, FacetBasis) and not hasattr(ubasis, "with_elements"):
            ubasis.with_elements = MethodType(with_elements, ubasis)

        self.ubasis = ubasis
        
        self.r = lob.shape[1]

        self.weight = np.array(elem_weight)
        self.nonzero_elements = np.nonzero(elem_weight)[0]

        self.ubasis_rom = ubasis.with_elements(self.nonzero_elements)


        # Mapping from full DOFs to indices in the reduced basis.
        self.mapping = self._get_mapping(self.ubasis_rom)
        self.element_dofs = self.ubasis_rom.element_dofs
        self.free_indices = self.mapping[self.element_dofs]
        self.mask = self.free_indices >= 0
        self.n_freedom = np.sum(self.mask, axis=0)


        self.unique_freedom = np.unique(self.n_freedom)

        self.cluster_idx=[]
        self.order_cluster=[]
        self.w_cluster=[]
        self.R_test_free=[]

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


            # col_idx = np.broadcast_to(np.arange(cluster_size), order.shape)


            # free_dofs_cluster: shape (cluster_size, nf); each row contains the local indices 
            # (in the element's DOF array) of the free DOFs.
            # free_dofs_cluster = fi_cluster[order, col_idx].T
            free_dofs_cluster = np.take_along_axis(fi_cluster, order, axis=0).T.astype(int)


            # Now, extract the rows from the projection matrices.
            # self.lob and self.rob are assumed to be arrays of shape (global_dofs, r).
            # Using free_dofs_cluster as indices produces:
            # R_test_free, R_trial_free: shape (cluster_size, nf, r)

            self.R_test_free.append(self.r_basis[free_dofs_cluster])


            # Gather the weights for the current cluster.
            if np.ndim(self.weight) == 0:
                self.w_cluster.append(self.weight * np.ones(cluster_size, dtype=self.dtype))

            else:
                # weight for restricted elements is weight[nonzero_elements]; then index by cluster_idx.
                self.w_cluster.append(self.weight[self.nonzero_elements][self.cluster_idx[-1]])


    def _get_mapping(self, basis: Basis) -> ndarray:
        """Build a mapping from global DOFs to reduced basis indices.
        
        TL;DR
        -----
        Build a mapping from global DOFs to reduced basis indices.
        
        Notes
        -----
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
        """Assemble the weighted reduced load vector.
        
        TL;DR
        -----
        Assemble the weighted reduced load vector.
        
        Notes
        -----
        Each element load vector is multiplied by its weight and projected onto
        the reduced basis (restricted to free DOFs), then summed into a single
        vector of length r.
        
        Parameters
        ----------
        **kwargs
            Additional parameters forwarded to `extract_element_vector_rom`, such
            as previous states or material parameters.
        
        Returns
        -------
        f_reduced : ndarray, shape (r,)
            Assembled reduced load vector.
        """
        
        

        # Extract the element load vectors for the restricted basis.
        element_vectors = self.extract_element_vector_rom(self.ubasis_rom, elem_indices=self.nonzero_elements, **kwargs)


        f_reduced = np.zeros((self.r,), dtype=self.dtype)
        

        i = 0 


        # Loop over clusters (the number of clusters is small, one per unique free DOF count).
        for nf in self.unique_freedom:


            # Extract the element vectors for the cluster.
            elem_vec_cluster = element_vectors[self.cluster_idx[i], :]


            # Use np.take_along_axis to pick the free entries.
            # self.order_cluster[i] should be an integer array of shape (cluster_size, nf)
            v_free_cluster = np.take_along_axis(elem_vec_cluster, self.order_cluster[i], axis=1)

            # Perform the vectorized projection. For each element, compute:
            #   v_red_element = R_test_free[e].T @ v_free_cluster[e]
            # Here, R_test_free[e] is (nf, r), so its transpose is (r, nf)
            # and the contraction over the nf index yields (r,)
            # Using einsum:


            cluster_contribution = np.einsum(
                'e, ern, en -> r',
                self.w_cluster[i],                      # shape (cluster_size,)
                self.R_test_free[i].transpose(0, 2, 1),  # shape (cluster_size, r, nf)
                v_free_cluster,                         # shape (cluster_size, nf)
                optimize=True
            )

            f_reduced += cluster_contribution

            i+=1

        return f_reduced
    

    def extract_element_vector_rom(self, basis: Basis, elem_indices: Optional[np.ndarray] = None, **kwargs):
        """Extract local element load vectors in the reduced setting.
        
        TL;DR
        -----
        Extract local element load vectors in the reduced setting.
        
        Notes
        -----
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
                """Assemble vector contributions inside a worker thread.
                
                TL;DR
                -----
                Assemble vector contributions inside a worker thread.
                
                Parameters
                ----------
                data : object
                    Value supplied as `data` for this helper.
                idx_chunk : object
                    Value supplied as `idx_chunk` for this helper.
                basis_list : object
                    Value supplied as `basis_list` for this helper.
                wdict : object
                    Value supplied as `wdict` for this helper.
                dx : object
                    Value supplied as `dx` for this helper.
                
                Returns
                -------
                None
                    This function updates state or performs work in place.
                
                Notes
                -----
                This helper is part of the surrounding workflow and keeps behavior local to the caller.
                """
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