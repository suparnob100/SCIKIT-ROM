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

    """Hyperreduction of a bilinear form via DEIM and mesh sampling.

    Implements a Discrete Empirical Interpolation Method (DEIM)–based
    hyperreduction for finite‐element bilinear forms. Builds a reduced‐order
    operator by assembling only a subset of elements and
    reconstructing the full operator via DEIM interpolation.
    """

    def __init__(self, form, elem_weight,
                 ubasis: Basis, lob, rob,
                 sampled_rows, deim_mat,
                 vbasis: Optional[Basis] = None,
                 free_dofs: Optional[ndarray] = None,
                 mean: Optional[ndarray] = None,
                 nthreads: int = 0,
                 dtype: DTypeLike = np.float64):
        
        """Parameters
        ----------
        form : callable
            The original bilinear form to be reduced (as in `BilinearForm`).
        elem_weight : array_like, shape (n_elements,)
            Element weights (1 for selected elements) indicating which elements participate in the reduced mesh.
            Zero weights drop elements from assembly.
        ubasis : Basis
            Trial/test basis for the full‐order model.
        lob, rob : ndarray, shape (n_free, r)
            Left (test) and right (trial) reduced bases. `rob` projects to the
            reduced trial space; `lob` is unused here (can pass `None`).
        sampled_rows : array_like, shape (n_samp,)
            Indices of global rows selected by DEIM for interpolation.
        deim_mat : ndarray, shape (r, n_samp)
            DEIM interpolation matrix mapping sampled DOFs back to the reduced basis.
        vbasis : Basis, optional
            Test basis for reduced assembly. Defaults to `ubasis` if not provided.
        free_dofs : ndarray, optional
            Indices of free (unconstrained) DOFs in the full‐order system.
        mean : ndarray, optional
            Mean snapshot vector for centering (if snapshot data is mean‐subtracted).
        nthreads : int, default 0
            Number of threads for parallel element‐matrix extraction; 0 means serial.
        dtype : DTypeLike, default np.float64
            NumPy data type for all internal arrays and computations.

        Attributes
        ----------
        weight : ndarray, shape (n_elements,)
            Element weights copy of `elem_weight`.
        nonzero_elements : ndarray, shape (n_active,)
            Indices of elements with nonzero weight.
        ubasis_rom : Basis
            Basis restricted to the hyperreduced mesh.
        sampled_rows : ndarray, shape (n_samp,)
            As above.
        n_samp : int
            Number of DEIM sample points.
        deim_mat : ndarray, shape (r, n_samp)
            As above.
        edofs : ndarray, shape (n_active, n_loc)
            Element‐to‐DOF mapping for restricted mesh.
        n_elems, n_loc : int
            Number of active elements and local DOFs per element.
        n_dofs : int
            Total number of global DOFs in the reduced mesh.
        rows, cols : ndarray
            Broadcasted element‐DOF indices for sparse assembly.
        row_flat, col_flat : ndarray
            Flattened indices for COO construction.
        """

        super().__init__(form)

        # ---------------- core variables ----------------
        self.rob = rob                  # right/trial basis (N_free × r)

        # ------------- sampled mesh --------
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
        Assemble the hyper-reduced stiffness matrix via DEIM.

        This first builds the sampled full-order matrix,
        then projects it onto the reduced basis using the DEIM
        interpolation matrix.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to `deim_elem_assembly`.

        Returns
        -------
        ndarray
            Reduced-order stiffness matrix of shape (r, r).
        """

        Sampled_assembly = self.deim_elem_assembly(**kwargs)
        Reduced_matrix = self.deim_mat @ Sampled_assembly[self.sampled_rows] @ self.rob

        return Reduced_matrix
    

    def deim_elem_assembly(self, **kwargs):

        """
        Assemble the sampled full-order stiffness matrix.

        Extracts element-level contributions only on sampled
        elements, flattens and filters out zeros, and builds
        a sparse CSR matrix.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to
            `extract_element_matrices_rom`.

        Returns
        -------
        csr_matrix
            Sparse full-order stiffness matrix of shape
            (n_dofs, n_dofs), assembled over sampled elements.
        """
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
        Extract element matrices for assembling over sampled mesh.

        Parameters
        ----------
        ubasis : Basis
            Basis for trial functions.
        vbasis : Basis, optional
            Basis for test functions. If None, `ubasis` is used.
        elem_indices : array_like of int, optional
            Indices of elements to include in extraction.
        **kwargs : dict
            Additional keyword arguments for evaluating the bilinear form over each element.

        Returns
        -------
        ndarray
            Array of shape (n_elems, n_loc, n_loc) containing
            the element stiffness matrices.
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