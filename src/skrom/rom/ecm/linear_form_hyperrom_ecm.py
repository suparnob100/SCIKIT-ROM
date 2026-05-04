"""ECM hyper-reduction for linear forms.

TL;DR
-----
This module assembles reduced load vectors using element and Gauss-point ECM weights.

Notes
-----
It applies weighted quadrature contributions and projects local vectors onto the reduced basis.
"""

from __future__ import annotations

from typing import Optional
from types import MethodType
from threading import Thread
import numpy as np
from numpy import ndarray
from numpy.typing import DTypeLike

from skfem.assembly.basis import Basis, FacetBasis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.linear_form import LinearForm

from .helpers import dense_ecm_weights, active_ecm_elements, with_elements


class LinearFormHYPERROM_ecm(LinearForm):
    """Hyperreduced linear form assembled from ECM-selected Gauss points.
    
    TL;DR
    -----
    Hyperreduced linear form assembled from ECM-selected Gauss points.
    """

    def __init__(
        self,
        form,
        gauss_weight,
        ubasis: Basis,
        lob,
        free_dofs: Optional[ndarray] = None,
        mean: Optional[ndarray] = None,
        nthreads: int = 0,
        dtype: DTypeLike = np.float64,
    ):
        """Initialize the LinearFormHYPERROM ECM instance.
        
        TL;DR
        -----
        Initialize the LinearFormHYPERROM ECM instance.
        
        Parameters
        ----------
        form : object
            Value supplied as `form` for this helper.
        gauss_weight : object
            Value supplied as `gauss_weight` for this helper.
        ubasis : object
            Value supplied as `ubasis` for this helper.
        lob : object
            Value supplied as `lob` for this helper.
        free_dofs : object
            Value supplied as `free_dofs` for this helper.
        mean : object
            Value supplied as `mean` for this helper.
        nthreads : object
            Value supplied as `nthreads` for this helper.
        dtype : object
            Value supplied as `dtype` for this helper.
        
        Returns
        -------
        None
            This function updates state or performs work in place.
        
        Notes
        -----
        This helper is part of the surrounding workflow and keeps behavior local to the caller.
        """
        super().__init__(form)

        self.r_basis = lob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype
        self.ubasis = ubasis
        self.r = lob.shape[1]

        if isinstance(ubasis, FacetBasis) and not hasattr(ubasis, "with_elements"):
            ubasis.with_elements = MethodType(with_elements, ubasis)

        self.n_elements = ubasis.nelems
        self.n_gauss_points = ubasis.dx.shape[1]
        self.gauss_weight = dense_ecm_weights(
            gauss_weight,
            self.n_elements,
            self.n_gauss_points,
            dtype=self.dtype,
        )
        self.nonzero_elements = active_ecm_elements(self.gauss_weight)

        self.ubasis_rom = ubasis.with_elements(self.nonzero_elements)
        self.gauss_weight_rom = self.gauss_weight[self.nonzero_elements]

        self.mapping = self._get_mapping(self.ubasis_rom)
        self.element_dofs = self.ubasis_rom.element_dofs
        self.free_indices = self.mapping[self.element_dofs]
        self.mask = self.free_indices >= 0
        self.n_freedom = np.sum(self.mask, axis=0)
        self.unique_freedom = np.unique(self.n_freedom)

        self.cluster_idx = []
        self.order_cluster = []
        self.w_cluster = []
        self.R_test_free = []

        for nf in self.unique_freedom:
            idx = np.nonzero(self.n_freedom == nf)[0]
            self.cluster_idx.append(idx)

            fm_cluster = self.mask[:, idx]
            fi_cluster = self.free_indices[:, idx]
            order = np.argsort(fm_cluster, axis=0)[-nf:, :]
            self.order_cluster.append(order.T)

            free_dofs_cluster = np.take_along_axis(fi_cluster, order, axis=0).T.astype(int)
            self.R_test_free.append(self.r_basis[free_dofs_cluster])
            self.w_cluster.append(self.gauss_weight_rom[idx])

    def _get_mapping(self, basis: Basis) -> ndarray:
        """Map global DOFs to indices used by the stored reduced basis.
        
        TL;DR
        -----
        Map global DOFs to indices used by the stored reduced basis.
        """
        n_full = basis.N
        if self.free_dofs is None:
            return np.arange(n_full, dtype=int)
        mapping = np.arange(n_full, dtype=int)
        non_free_dofs = np.setdiff1d(mapping, self.free_dofs)
        mapping[non_free_dofs] = -1
        return mapping

    def assemble_weighted_ecm(self, **kwargs) -> ndarray:
        """Assemble the reduced linear form using ECM Gauss-point weights.
        
        TL;DR
        -----
        Assemble the reduced linear form using ECM Gauss-point weights.
        """
        element_vectors_q = self.extract_element_vectors_qp_rom(
            self.ubasis_rom,
            elem_indices=self.nonzero_elements,
            **kwargs,
        )

        f_reduced = np.zeros((self.r,), dtype=self.dtype)

        for ic, nf in enumerate(self.unique_freedom):
            elem_vec_cluster = element_vectors_q[self.cluster_idx[ic]]  # (e, nloc, nq)
            order = self.order_cluster[ic]
            v_free_cluster = np.take_along_axis(elem_vec_cluster, order[:, :, None], axis=1)

            cluster_contribution = np.einsum(
                'eia,eaq,eq->i',
                self.R_test_free[ic].transpose(0, 2, 1),
                v_free_cluster,
                self.w_cluster[ic],
                optimize=True,
            )
            f_reduced += cluster_contribution

        return f_reduced

    def _kernel_qp(self, v, w, dx) -> ndarray:
        """Return per-element, per-Gauss local vector contributions.
        
        TL;DR
        -----
        Return per-element, per-Gauss local vector contributions.
        """
        val = np.asarray(self.form(*v, w) * dx, dtype=self.dtype)
        if val.ndim != 2:
            raise ValueError(
                "ECM assembly expects linear forms to return a 2D array of "
                "shape (n_elements, n_gauss_points) after multiplying by dx."
            )
        if val.shape == dx.shape:
            return val
        if val.shape == dx.T.shape:
            return val.T
        raise ValueError(
            "Unexpected linear kernel shape. "
            f"Got {val.shape}; expected {dx.shape} or {dx.T.shape}."
        )

    def _threaded_kernel_qp(self, data, ix, basis_list, wdict, dx):
        """Assemble quadrature-point contributions inside a worker thread.
        
        TL;DR
        -----
        Assemble quadrature-point contributions inside a worker thread.
        
        Parameters
        ----------
        data : object
            Value supplied as `data` for this helper.
        ix : object
            Value supplied as `ix` for this helper.
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
        for i in ix:
            data[i] = self._kernel_qp(basis_list[i], wdict, dx)

    def extract_element_vectors_qp_rom(
        self,
        basis: Basis,
        elem_indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ndarray:
        """Extract per-Gauss local element vectors.
        
        TL;DR
        -----
        Extract per-Gauss local element vectors.
        
        Returns
        -------
        element_vectors_q : ndarray, shape (n_elem, n_loc, n_q)
            Per-element local vectors resolved at Gauss points, already multiplied
            by ``basis.dx`` but not yet summed over Gauss points.
        """
        nt = basis.nelems
        dx = basis.dx
        nq = dx.shape[1]

        wdict = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })
        wdict['elem_indices'] = elem_indices

        local_data = np.zeros((basis.Nbfun, nt, nq), dtype=self.dtype)

        if self.nthreads <= 0:
            for i in range(basis.Nbfun):
                local_data[i, :, :] = self._kernel_qp(basis.basis[i], wdict, dx)
        else:
            indices = np.arange(basis.Nbfun)
            threads = [
                Thread(
                    target=self._threaded_kernel_qp,
                    args=(local_data, ix, basis.basis, wdict, dx),
                )
                for ix in np.array_split(indices, self.nthreads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        return np.transpose(local_data, (1, 0, 2))
