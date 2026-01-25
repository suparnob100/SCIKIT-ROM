"""
Linear form placeholder.

The vibration example solves a generalized eigenproblem, so there is no load vector.
This file is kept to match the example structure used across the repository.
"""

from skfem import LinearForm


@LinearForm
def l_zero(v, w):
    """Return zero load."""
    return 0.0 * v
