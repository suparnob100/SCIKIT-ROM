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

    """Hyperreduction of a linear form via DEIM and mesh sampling.

    Implements a Discrete Empirical Interpolation Method (DEIM)–based
    hyperreduction for finite‐element linear forms. Builds a reduced‐order
    load vector by assembling only a weighted subset of elements and
    reconstructing the full operator via DEIM interpolation.
    """

    def __init__(self, form, elem_weight, ubasis: Basis, lob,
                 sampled_rows, deim_mat,
                 free_dofs: Optional[np.ndarray] = None,
                 mean: Optional[np.ndarray] = None,
                 nthreads=0, dtype=np.float64):

        super().__init__(form)

        """
        Parameters
        ----------
        form : callable
            The original linear form to be reduced (as in `LinearForm`).
        elem_weight : array_like, shape (n_elements,)
            Element weights indicating which elements participate in the reduced mesh.
            Zero weights drop elements from assembly.
        ubasis : Basis
            Test basis for the full‐order model.
        lob : ndarray, shape (n_free, r)
            Left (test) reduced basis. Projects to the reduced test space.
        sampled_rows : array_like, shape (n_samp,)
            Indices of global rows selected by DEIM for interpolation.
        deim_mat : ndarray, shape (r, n_samp)
            DEIM interpolation matrix mapping sampled DOFs back to the reduced basis.
        free_dofs : ndarray, optional
            Indices of free (unconstrained) DOFs in the full‐order system.
        mean : ndarray, optional
            Mean snapshot vector for centering (if snapshot data is mean‐subtracted).
        nthreads : int, default 0
            Number of threads for parallel element‐vector extraction; 0 means serial.
        dtype : DTypeLike, default np.float64
            NumPy data type for all internal arrays and computations.

        Attributes
        ----------
        r_basis : ndarray, shape (n_free, r)
            Left (test) reduced basis copy of `lob`.
        weight : ndarray, shape (n_elements,)
            Element weights copy of `elem_weight`.
        nonzero_elements : ndarray, shape (n_active,)
            Indices of elements with nonzero weight.
        ubasis : Basis
            Original full‐order basis reference.
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
        n_dofs : int
            Total number of global DOFs in the reduced mesh.
        rows : ndarray, shape (n_active * n_loc,)
            Flattened element‐DOF indices for vector assembly.

        """
        
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
        Assemble the hyper-reduced load vector via DEIM.

        This method first builds the sampled full-order load vector
        on the selected elements, then applies the DEIM interpolation
        matrix to project it onto the reduced basis.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to
            `deim_elem_assembly` / element extraction.

        Returns
        -------
        ndarray
            Reduced-order load vector of shape (r,).
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
        """
        Assemble the sampled full-order load vector.

        Extracts element-level load contributions only on the
        sampled elements, then scatters them into the global
        load vector.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to
            `extract_element_vector_rom`.

        Returns
        -------
        ndarray
            Full-order load vector of length `n_dofs`.
        """

        element_vectors = self.extract_element_vector_rom(self.ubasis_rom, elem_indices=self.nonzero_elements, **kwargs)
        data = element_vectors.T.ravel()       # same length, in matching order

        # add contributions at once
        f = np.zeros(self.n_dofs)
        np.add.at(f, self.rows, data)

        return f


    def extract_element_vector_rom(self, basis: Basis, elem_indices = None, **kwargs):
        """
        Extract element vectors for assembling over sampled mesh.

        Parameters
        ----------
        ubasis : Basis
            Basis for test functions.
        elem_indices : array_like of int, optional
            Indices of elements to include in extraction.
        **kwargs : dict
            Additional keyword arguments for evaluating the bilinear form over each element.

        Returns
        -------
        ndarray
            Array of shape (n_elems, n_loc) containing
            the element vectors.
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
  