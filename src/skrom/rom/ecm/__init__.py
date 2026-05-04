"""ECM hyper-reduction package.

TL;DR
-----
This package exposes Empirical Cubature Method helpers and form assemblers.

Notes
-----
It connects dense ECM weights, active element lookup, and weighted bilinear or linear assembly classes.
"""

from .helpers import (
    dense_ecm_weights,
    flat_to_element_gauss_weights,
    active_ecm_elements,
)
from .bilinear_form_hyperrom_ecm import BilinearFormHYPERROM_ecm
from .linear_form_hyperrom_ecm import LinearFormHYPERROM_ecm

__all__ = [
    "dense_ecm_weights",
    "flat_to_element_gauss_weights",
    "active_ecm_elements",
    "BilinearFormHYPERROM_ecm",
    "LinearFormHYPERROM_ecm",
]
