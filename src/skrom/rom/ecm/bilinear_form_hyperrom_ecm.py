"""ECM hyper-reduction for bilinear forms.

TL;DR
-----
This module assembles reduced stiffness matrices using element and Gauss-point ECM weights.

Notes
-----
It clusters active elements, applies weighted quadrature contributions, and projects local matrices onto the reduced basis.
"""

from __future__ import annotations

from typing import Optional, Any
from types import MethodType
from threading import Thread
import numpy as np
from numpy import ndarray
from numpy.typing import DTypeLike

from skfem.assembly.basis import Basis, FacetBasis
from skfem.assembly.form.form import FormExtraParams
from skfem.assembly.form.bilinear_form import BilinearForm

from .helpers import dense_ecm_weights, active_ecm_elements, with_elements


class BilinearFormHYPERROM_ecm(BilinearForm):
    """Hyperreduced bilinear form assembled from ECM-selected Gauss points.
    
    TL;DR
    -----
    Hyperreduced bilinear form assembled from ECM-selected Gauss points.
    """

    def __init__(
        self,
        form,
        gauss_weight,
        ubasis: Basis,
        lob,
        rob,
        vbasis: Optional[Basis] = None,
        free_dofs: Optional[ndarray] = None,
        mean: Optional[ndarray] = None,
        nthreads: int = 0,
        dtype: DTypeLike = np.float64,
    ):
        """Initialize the BilinearFormHYPERROM ECM instance.
        
        TL;DR
        -----
        Initialize the BilinearFormHYPERROM ECM instance.
        
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
        rob : object
            Value supplied as `rob` for this helper.
        vbasis : object
            Value supplied as `vbasis` for this helper.
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

        self.lob = lob
        self.rob = rob
        self.free_dofs = free_dofs
        self.mean = mean
        self.nthreads = nthreads
        self.dtype = dtype
        self.ubasis = ubasis
        self.vbasis = ubasis if vbasis is None else vbasis
        self.r = lob.shape[1]

        if isinstance(ubasis, FacetBasis) and not hasattr(ubasis, "with_elements"):
            ubasis.with_elements = MethodType(with_elements, ubasis)
        if isinstance(self.vbasis, FacetBasis) and not hasattr(self.vbasis, "with_elements"):
            self.vbasis.with_elements = MethodType(with_elements, self.vbasis)

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
        self.vbasis_rom = self.vbasis.with_elements(self.nonzero_elements)
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
        self.R_trial_free = []

        for nf in self.unique_freedom:
            idx = np.nonzero(self.n_freedom == nf)[0]
            self.cluster_idx.append(idx)

            fm_cluster = self.mask[:, idx]
            fi_cluster = self.free_indices[:, idx]
            order = np.argsort(fm_cluster, axis=0)[-nf:, :]
            self.order_cluster.append(order.T)

            free_dofs_cluster = np.take_along_axis(fi_cluster, order, axis=0).T.astype(int)
            self.R_test_free.append(self.lob[free_dofs_cluster])
            self.R_trial_free.append(self.rob[free_dofs_cluster])
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
        """Assemble the reduced bilinear form using ECM Gauss-point weights.
        
        TL;DR
        -----
        Assemble the reduced bilinear form using ECM Gauss-point weights.
        """
        element_matrices_q = self.extract_element_matrices_qp_rom(
            self.ubasis_rom,
            self.vbasis_rom,
            elem_indices=self.nonzero_elements,
            **kwargs,
        )

        k_reduced = np.zeros((self.r, self.r), dtype=self.dtype)

        for ic, nf in enumerate(self.unique_freedom):
            elem_mat_cluster = element_matrices_q[self.cluster_idx[ic]]  # (e, nloc, nloc, nq)
            order = self.order_cluster[ic]

            k_rows = np.take_along_axis(elem_mat_cluster, order[:, :, None, None], axis=1)
            k_cluster_free = np.take_along_axis(k_rows, order[:, None, :, None], axis=2)

            cluster_contribution = np.einsum(
                'eia,eabq,eq,ebj->ij',
                self.R_test_free[ic].transpose(0, 2, 1),
                k_cluster_free,
                self.w_cluster[ic],
                self.R_trial_free[ic],
                optimize=True,
            )
            k_reduced += cluster_contribution

        return k_reduced

    def _kernel_qp(self, u, v, w, dx) -> ndarray:
        """Return per-element, per-Gauss contributions before quadrature summation.
        
        TL;DR
        -----
        Return per-element, per-Gauss contributions before quadrature summation.
        """
        val = np.asarray(self.form(*u, *v, w) * dx, dtype=self.dtype)
        if val.ndim != 2:
            raise ValueError(
                "ECM assembly expects bilinear forms to return a 2D array of "
                "shape (n_elements, n_gauss_points) after multiplying by dx."
            )
        if val.shape == dx.shape:
            return val
        if val.shape == dx.T.shape:
            return val.T
        raise ValueError(
            "Unexpected bilinear kernel shape. "
            f"Got {val.shape}; expected {dx.shape} or {dx.T.shape}."
        )

    def _threaded_kernel_qp(self, data, ix, ubasis, vbasis, wdict, dx):
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
        ubasis : object
            Value supplied as `ubasis` for this helper.
        vbasis : object
            Value supplied as `vbasis` for this helper.
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
        for ij in ix:
            i, j = ij
            data[j, i] = self._kernel_qp(ubasis[j], vbasis[i], wdict, dx)

    def extract_element_matrices_qp_rom(
        self,
        ubasis: Basis,
        vbasis: Optional[Basis] = None,
        elem_indices: Optional[ndarray] = None,
        **kwargs,
    ) -> ndarray:
        """Extract per-Gauss local element matrices.
        
        TL;DR
        -----
        Extract per-Gauss local element matrices.
        
        Returns
        -------
        element_matrices_q : ndarray, shape (n_elem, n_loc_test, n_loc_trial, n_q)
            Per-element local matrices resolved at Gauss points, already multiplied
            by ``basis.dx`` but not yet summed over Gauss points.
        """
        if vbasis is None:
            vbasis = ubasis

        nt = ubasis.nelems
        dx = ubasis.dx
        nq = dx.shape[1]
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })
        wdict['elem_indices'] = elem_indices

        local_data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt, nq), dtype=self.dtype)

        if self.nthreads <= 0:
            for j in range(ubasis.Nbfun):
                for i in range(vbasis.Nbfun):
                    local_data[j, i, :, :] = self._kernel_qp(
                        ubasis.basis[j],
                        vbasis.basis[i],
                        wdict,
                        dx,
                    )
        else:
            indices = np.array([[i, j] for j in range(ubasis.Nbfun) for i in range(vbasis.Nbfun)])
            threads = [
                Thread(
                    target=self._threaded_kernel_qp,
                    args=(local_data, ix, ubasis.basis, vbasis.basis, wdict, dx),
                )
                for ix in np.array_split(indices, self.nthreads, axis=0)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        return np.transpose(local_data, (2, 1, 0, 3))
