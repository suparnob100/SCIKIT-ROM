import numpy as np
import matplotlib.pyplot as plt

from skrom.rom.ecm.helpers import flat_to_element_gauss_weights
from skrom.rom.rom_utils import collect_residuals_ecm
from skrom.rom.ecm.empirical_cubature_method import EmpiricalCubatureMethod
from sklearn.utils.extmath import randomized_svd


def ECM(
    q_mus_ecm,
    n_elements,
    n_gauss_points,
    tol=1e-5,
    n_components=None,
    plot=False,
    constrain_sum_of_weights=False,
    dtype=np.float64,
    svd_rank=None
):
    """
    ECM adapted to scikit-rom.

    Parameters
    ----------
    q_mus_ecm : ndarray, shape (n_snapshots * r, n_elements * n_gauss_points)
        Matrix returned by collect_residuals_ecm(...).
        Each column corresponds to one element-Gauss-point contribution.
    n_elements : int
        Number of elements in the mesh.
    n_gauss_points : int
        Number of Gauss points per element.
    tol : float, optional
        Singular-value cutoff used to choose the ECM basis size.
    n_components : int, optional
        If given, overrides tol and uses exactly this many left singular vectors.
    plot : bool, optional
        If True, plots the singular values.
    constrain_sum_of_weights : bool, optional
        Passed to EmpiricalCubatureMethod.SetUp(...).
    dtype : dtype, optional
        Floating-point dtype.

    Returns
    -------
    W : ndarray, shape (n_selected,)
        ECM weights for the selected flat indices.
    Z : ndarray, shape (n_selected,)
        Selected flat indices in [0, n_elements * n_gauss_points).
    S : ndarray
        Singular values of q_mus_ecm.T.
    gauss_weight_ecm : ndarray, shape (n_elements, n_gauss_points)
        Dense ECM multiplier array for online assembly in scikit-rom.
    """
    q_mus_ecm = np.asarray(q_mus_ecm, dtype=dtype)

    if q_mus_ecm.ndim != 2:
        raise ValueError("q_mus_ecm must be a 2D array.")

    n_rows_expected = n_elements * n_gauss_points
    if q_mus_ecm.shape[1] != n_rows_expected:
        raise ValueError(
            f"q_mus_ecm has {q_mus_ecm.shape[1]} columns, but "
            f"n_elements * n_gauss_points = {n_rows_expected}."
        )

    # Original external ECM code assembled R_FE with shape
    # (n_elements * n_gauss_points, n_snapshots * r).
    # collect_residuals_ecm returns its transpose.
    R_FE = q_mus_ecm.T.copy()

        # SVD of the ECM training matrix
    if svd_rank is None:
        U_FE, S, _ = np.linalg.svd(R_FE, full_matrices=False)
    else:
        U_FE, S, _ = randomized_svd(
            R_FE,
            n_components=min(svd_rank, min(R_FE.shape)),
            n_oversamples=10,
            n_iter="auto",
            random_state=0,
        )
    
    if plot:
        plt.figure()
        plt.semilogy(S, "o-")
        plt.xlabel("Index")
        plt.ylabel("Singular value")
        plt.title("ECM singular values")
        plt.grid(True)
        plt.show()

    if n_components is None:
        cutoff = 1e-5 if tol is None else tol
        below = np.flatnonzero(S < cutoff)
        if below.size > 0:
            n_components = max(1, int(below[0]))
        else:
            n_components = len(S)

    n_components = min(int(n_components), U_FE.shape[1])
    if n_components < 1:
        raise ValueError("n_components must be at least 1.")

    # IMPORTANT:
    # q_mus_ecm already contains contributions multiplied by basis.dx.
    # So for scikit-rom ECM assembly, we use unit row weights here.
    # Then the returned dense weights act directly as multipliers
    # on the pre-weighted quadrature contributions.
    row_weights = np.ones(n_rows_expected, dtype=dtype)

    ecm_solver = EmpiricalCubatureMethod()
    ecm_solver.SetUp(
        U_FE[:, :n_components].T,
        Weights=row_weights,
        constrain_sum_of_weights=constrain_sum_of_weights,
    )
    ecm_solver.Run()

    W = np.asarray(np.squeeze(ecm_solver.w), dtype=dtype)
    Z = np.asarray(ecm_solver.z, dtype=int).ravel()

    gauss_weight_ecm = flat_to_element_gauss_weights(
        n_elements=n_elements,
        n_gauss_points=n_gauss_points,
        selected_indices=Z,
        selected_weights=W,
        base_quadrature_weights=None,   # already multiplier weights
        dtype=dtype,
    )

    return W, Z, S, gauss_weight_ecm


def ECM_from_skrom(
    NLS_train_ms,
    NLS_train_mean,
    V_sel,
    reconstruct_solution,
    Residual,
    training_params,
    assemble_kwargs,
    basis,
    *,
    extra_kwargs=None,
    hyper_basis=None,
    tol=1e-5,
    n_components=None,
    plot=False,
    constrain_sum_of_weights=False,
    dtype=np.float64,
    svd_rank=None
):
    """
    End-to-end ECM wrapper for the scikit-rom notebook structure.

    Returns
    -------
    result : dict
        Keys:
        - q_mus_ecm
        - W
        - Z
        - S
        - gauss_weight_ecm
    """
    q_mus_ecm = collect_residuals_ecm(
        NLS_train_ms,
        NLS_train_mean,
        V_sel,
        reconstruct_solution,
        Residual,
        training_params,
        assemble_kwargs,
        extra_kwargs=extra_kwargs,
        hyper_basis=hyper_basis,
    )

    n_elements = basis.nelems
    n_gauss_points = basis.dx.shape[1]

    W, Z, S, gauss_weight_ecm = ECM(
        q_mus_ecm=q_mus_ecm,
        n_elements=n_elements,
        n_gauss_points=n_gauss_points,
        tol=tol,
        n_components=n_components,
        plot=plot,
        constrain_sum_of_weights=constrain_sum_of_weights,
        dtype=dtype,
        svd_rank=svd_rank
    )

    return {
        "q_mus_ecm": q_mus_ecm,
        "W": W,
        "Z": Z,
        "S": S,
        "gauss_weight_ecm": gauss_weight_ecm,
    }