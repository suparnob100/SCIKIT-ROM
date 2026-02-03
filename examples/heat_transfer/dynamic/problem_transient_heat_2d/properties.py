"""\
Parameter-dependent coefficients for 2-D transient heat conduction.

Parameters
----------
param[0] : k  (thermal conductivity scaling)
param[1] : q  (source amplitude scaling)

The returned callables follow the (value, region) signature used in other examples.
"""

from __future__ import annotations


def k(k_param: float, region: str = "region_1") -> float:
    """Thermal conductivity scaling."""
    return float(k_param)


def q(q_param: float, region: str = "region_1") -> float:
    """Heat-source amplitude scaling."""
    return float(q_param)
