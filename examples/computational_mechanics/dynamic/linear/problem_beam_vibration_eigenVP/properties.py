"""
Section properties and coefficient maps for the Timoshenko beam.

Parameters passed by the sampler:
- h1: thickness of region_1
- h2: thickness of region_2

Fixed constants (edit as needed):
- E: Young's modulus
- nu: Poisson ratio
- rho: density
- b: width
- kappa: shear correction factor
"""

import numpy as np


# Fixed constants (can be edited)
E_DEFAULT = 210e9       # Pa
NU_DEFAULT = 0.30
RHO_DEFAULT = 7800.0    # kg/m^3
B_DEFAULT = 0.05        # m
KAPPA_DEFAULT = 5.0 / 6.0


def thickness(h1: float, h2: float, region: str) -> float:
    """Return segment thickness for a region."""
    if region == "region_1":
        return float(h1)
    if region == "region_2":
        return float(h2)
    raise KeyError(f"Unknown region: {region}")


def beam_coeffs(
    h1: float,
    h2: float,
    region: str,
    E: float = E_DEFAULT,
    nu: float = NU_DEFAULT,
    rho: float = RHO_DEFAULT,
    b: float = B_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
):
    """
    Compute scalar coefficients used in affine operators.

    Returns
    -------
    EI : float
        Bending coefficient E*I
    kGA : float
        Shear coefficient kappa*G*A
    rhoA : float
        Translational inertia per length
    rhoI : float
        Rotational inertia per length
    """
    h = thickness(h1, h2, region)
    A = b * h
    I = b * h**3 / 12.0
    G = E / (2.0 * (1.0 + nu))

    EI = E * I
    kGA = kappa * G * A
    rhoA = rho * A
    rhoI = rho * I

    return EI, kGA, rhoA, rhoI
