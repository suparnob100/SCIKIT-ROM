"""Reduced-order modeling package.

TL;DR
-----
This package collects projection, error analysis, sampling, and hyper-reduction tools for ROM workflows.

Notes
-----
It exposes ECM helpers and groups the core ROM, DEIM, ECSW, and ECM implementations.
"""

from .ecm import (
    dense_ecm_weights,
    flat_to_element_gauss_weights,
    active_ecm_elements,
    BilinearFormHYPERROM_ecm,
    LinearFormHYPERROM_ecm,
)
