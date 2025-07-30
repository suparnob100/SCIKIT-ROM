from skfem.models.elasticity import lame_parameters

"""
Module for region-specific Lamé parameter adjustments.
"""

def lame_params(E, nu, region):
    """
    Compute Lamé parameters (λ, μ) for a given region.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    region : str
        Identifier for the subdomain (e.g., 'region_1', 'region_2').

    Returns
    -------
    lam : float
        First Lamé parameter, possibly scaled by region.
    mu : float
        Second Lamé parameter, possibly scaled by region.

    Notes
    -----
    - For 'region_1', both λ and μ are scaled by a factor of 50.
    - For 'region_2', λ and μ are returned unchanged.
    - No other regions are currently handled explicitly.
    """
    # Base parameters from SKFE elasticity model
    lam, mu = lame_parameters(E, nu)

    if region == 'region_1':
        # Material properties scaled in region 1
        return 50.0 * lam, 50.0 * mu
    elif region == 'region_2':
        # Default properties in region 2
        return lam, mu
    else:
        # Fallback for unspecified regions
        return lam, mu