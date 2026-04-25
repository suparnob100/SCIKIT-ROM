from skfem.models.elasticity import lame_parameters

"""
Material properties for plane stress isotropic linear elasticity.
"""

def lame_params_plane_stress(E, nu, region=None):
    """
    Return effective Lamé-like coefficients for a 2D plane-stress formulation.

    We start from the 3D Lamé parameters (λ, μ) and eliminate σ_zz=0, which yields
    an in-plane constitutive law that is equivalent to using:

        λ_bar = 2 μ λ / (λ + 2 μ)
        μ     = μ

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    region : str, optional
        Included for interface consistency (unused here).

    Returns
    -------
    lam_bar : float
        Effective first coefficient for plane stress.
    mu : float
        Shear modulus.
    """
    lam, mu = lame_parameters(E, nu)
    lam_bar = (2.0 * mu * lam) / (lam + 2.0 * mu)
    return lam_bar, mu
