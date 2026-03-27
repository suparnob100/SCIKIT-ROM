"""
Helper utilities for ECM-based hyperreduction.

This module converts ECM-selected Gauss-point data into dense per-element,
per-Gauss-point arrays suitable for ROM assembly.

Notes
-----
The bilinear and linear ECM assemblers in this folder assume that the stored
ECM weights act as multiplicative factors on scikit-fem's native quadrature
contributions, i.e. on quantities that already include ``basis.dx``.

If an external ECM implementation returns *absolute* cubature weights for the
reference quadrature rule, pass the reference Gauss weights through
``base_quadrature_weights`` so that the helper divides by them and produces the
multipliers expected by the assemblers.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Any
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from skfem.assembly.basis import FacetBasis


def with_elements(self, elements: Any = None) -> FacetBasis:
    """Return a similar ``FacetBasis`` restricted to a subset of elements."""
    return type(self)(
        self.mesh,
        self.elem,
        mapping=self.mapping,
        quadrature=self.quadrature,
        facets=elements,
    )



def _as_2d_array(data: ArrayLike, n_elements: int, n_gauss_points: int, dtype: DTypeLike) -> np.ndarray:
    """Convert supported dense ECM-weight inputs to shape ``(n_elements, n_gauss_points)``."""
    arr = np.asarray(data, dtype=dtype)

    if arr.ndim == 0:
        return np.full((n_elements, n_gauss_points), arr.item(), dtype=dtype)

    if arr.ndim == 2:
        if arr.shape != (n_elements, n_gauss_points):
            raise ValueError(
                "Dense ECM weights must have shape "
                f"({n_elements}, {n_gauss_points}), got {arr.shape}."
            )
        return arr.copy()

    if arr.ndim == 1:
        if arr.size == n_elements * n_gauss_points:
            return arr.reshape(n_elements, n_gauss_points).copy()
        raise ValueError(
            "Flat ECM weights must have length n_elements * n_gauss_points; "
            f"got {arr.size}."
        )

    raise ValueError("Unsupported ECM weight format. Use dict, 2D array, or flat 1D array.")



def dense_ecm_weights(
    weight_data: Mapping[int, Sequence[float]] | ArrayLike,
    n_elements: int,
    n_gauss_points: int,
    *,
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """
    Build a dense ECM-weight array of shape ``(n_elements, n_gauss_points)``.

    Parameters
    ----------
    weight_data
        Either:
        - a mapping ``element_id -> sequence_of_length_n_gauss_points``; or
        - a dense array of shape ``(n_elements, n_gauss_points)``; or
        - a flat array of length ``n_elements * n_gauss_points``.
    n_elements
        Number of elements in the full mesh.
    n_gauss_points
        Number of Gauss points per element.
    dtype
        Numeric dtype of the returned array.
    """
    if isinstance(weight_data, Mapping):
        dense = np.zeros((n_elements, n_gauss_points), dtype=dtype)
        for e, weights in weight_data.items():
            if e < 0 or e >= n_elements:
                raise ValueError(f"Element index {e} is out of range [0, {n_elements}).")
            row = np.asarray(weights, dtype=dtype).ravel()
            if row.size != n_gauss_points:
                raise ValueError(
                    f"Element {e} has {row.size} Gauss weights; expected {n_gauss_points}."
                )
            dense[e, :] = row
        return dense

    return _as_2d_array(weight_data, n_elements, n_gauss_points, dtype)



def flat_to_element_gauss_weights(
    n_elements: int,
    n_gauss_points: int,
    selected_indices: ArrayLike,
    selected_weights: ArrayLike,
    *,
    base_quadrature_weights: ArrayLike | None = None,
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """
    Convert flat ECM-selected Gauss-point data to a dense element/Gauss array.

    Parameters
    ----------
    n_elements
        Number of elements in the full mesh.
    n_gauss_points
        Number of Gauss points per element.
    selected_indices
        Flat Gauss-point indices as returned by an ECM solve.
    selected_weights
        ECM weights aligned with ``selected_indices``.
    base_quadrature_weights
        Optional base quadrature weights of length ``n_gauss_points``.
        When supplied, the returned values are divided by these weights so that
        the result acts as a multiplicative factor on scikit-fem's native
        ``basis.dx`` contributions.
    dtype
        Numeric dtype of the returned array.
    """
    idx = np.asarray(selected_indices, dtype=int).ravel()
    w = np.asarray(selected_weights, dtype=dtype).ravel()
    if idx.size != w.size:
        raise ValueError("selected_indices and selected_weights must have the same length.")

    dense = np.zeros((n_elements, n_gauss_points), dtype=dtype)
    max_flat = n_elements * n_gauss_points
    for flat_idx, weight in zip(idx, w):
        if flat_idx < 0 or flat_idx >= max_flat:
            raise ValueError(f"Flat Gauss-point index {flat_idx} is out of range [0, {max_flat}).")
        e = flat_idx // n_gauss_points
        q = flat_idx % n_gauss_points
        dense[e, q] += weight

    if base_quadrature_weights is not None:
        qref = np.asarray(base_quadrature_weights, dtype=dtype).ravel()
        if qref.size != n_gauss_points:
            raise ValueError(
                f"base_quadrature_weights must have length {n_gauss_points}; got {qref.size}."
            )
        with np.errstate(divide='ignore', invalid='ignore'):
            dense = np.divide(dense, qref[None, :], out=np.zeros_like(dense), where=qref[None, :] != 0)

    return dense



def active_ecm_elements(weight_array: ArrayLike, *, atol: float = 0.0) -> np.ndarray:
    """Return element indices whose ECM weight row has at least one active Gauss point."""
    arr = np.asarray(weight_array)
    if arr.ndim != 2:
        raise ValueError("weight_array must be 2D with shape (n_elements, n_gauss_points).")
    return np.flatnonzero(np.any(np.abs(arr) > atol, axis=1))
