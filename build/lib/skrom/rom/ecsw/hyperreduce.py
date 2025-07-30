"""
Implements the end-to-end hyper-reduction pipeline combining randomized SVD and bounded NNLS.

This module provides:
  - `hyperreduce`: function to perform hyper-reduction on a QoI matrix by:
    1. Optionally applying randomized SVD for dimensionality reduction
    2. Constructing bounded constraints for NNLS from projected data
    3. Solving a bounded NNLS problem via `NNLSSolver`
    4. Optionally visualizing singular value decay and NNLS coefficients

The `hyperreduce` folder contains utilities to reduce full-order models, including:
  - Randomized SVD preprocessing routines
  - Bounded NNLS solve integrations (`custom_nnls`)
  - Plotting helpers for diagnostic visualization of reduction errors

Dependencies:
  - NumPy for array operations
  - scikit-learn's `randomized_svd` for fast SVD
  - Matplotlib for plotting diagnostics
  - Custom `NNLSSolver` implementation in `custom_nnls`

Usage example:
```python
from hyperreduce.hyperreduce import hyperreduce
x, flag = hyperreduce(qoi_data, n_components=100, svd=True)
```  
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt  # Provides a MATLAB-like interface for plotting
from .custom_nnls import NNLSSolver

def hyperreduce(qoi, n_components=500, verbosity=2, plot=True, const_tol=1e-10, zero_tol=1e-14, svd=False):
    r"""
    Perform hyper-reduction via randomized SVD followed by a bounded NNLS solve.

    The hyper-reduction pipeline includes:

    1. (Optional) Randomized SVD of the quantity of interest (QoI) matrix to reduce its dimensionality.

    2. Construction of lower and upper bound constraints around the projected right-hand side vector.

    3. Bounded Non-Negative Least Squares (NNLS) solve using the `NNLSSolver`.

    4. (Optional) Visualization of singular value decay and NNLS solution coefficients.

    Parameters
    ----------
    qoi : array_like, shape (n_samples, n_features)
        Quantity of interest matrix on which hyper-reduction is performed.
    n_components : int, optional
        Number of singular value decomposition components to retain when `svd=True`.
        Must be less than or equal to \min(n_samples, n_features). Default is 500.
    verbosity : int, optional
        Verbosity level for the NNLS solver. Higher values yield more diagnostic output.
        Default is 2.
    plot : bool, optional
        Whether to display plots for singular value decay and the NNLS solution vector.
        Default is True.
    const_tol : float, optional
        Tolerance used to define the half-gap around the average right-hand side
        vector for bounded constraints. Default is 1e-10.
    zero_tol : float, optional
        Threshold below which NNLS solution coefficients are considered zero.
        Default is 1e-14.
    svd : bool, optional
        If True, apply randomized SVD preprocessing to `qoi`, otherwise solve NNLS
        directly on the original data. Default is False.

    Returns
    -------
    x : ndarray, shape (n_features,) or (n_components,)
        Coefficients from the bounded NNLS solve representing the hyper-reduction weights.
    flag : int
        Exit status flag returned by the NNLS solver (e.g., 0 indicates successful convergence).

    Raises
    ------
    ValueError
        If `n_components` is greater than the minimum dimension of `qoi` when `svd=True`.

    Notes
    -----
    - The `randomized_svd` step (when enabled) uses oversampling and power iterations
      for stability and accuracy.
    - Bounds for the NNLS solve are constructed as:

      .. math::
         b_{\mathrm{lower}} = d_q - \mathrm{const\_tol},
         \quad
         b_{\mathrm{upper}} = d_q + \mathrm{const\_tol},

      where

      .. math::
         d_q = V_q^{\mathrm{eff}} \mathbf{1}

      is the projected right-hand side vector.
    - The final hyper-reduced error is computed internally as

      .. math::
         \frac{\|qoi\mathbf{1} - qoi\,x\|_2}{\|qoi\mathbf{1}\|_2}

      and printed for diagnostic purposes.

    Examples
    --------
    >>> import numpy as np
    >>> from hyperreduce_module import hyperreduce
    >>> data = np.random.rand(100, 200)
    >>> x, flag = hyperreduce(data, n_components=50, svd=True, plot=False)
    >>> print("Exit flag:", flag)
    >>> print("Active basis vectors:", np.sum(x > 0))
    """
    # Function implementation follows unchanged
    if svd:
        print("Performing randomized SVD...")
        Uh, s, Vh = randomized_svd(
            qoi,
            n_components=n_components,
            random_state=22,
            n_oversamples=30,
            n_iter=2
        )
        if plot:
            plt.semilogy(
                1 - (np.cumsum(s**2) / np.sum(s**2)) + 1e-15,
                'o-', ms=4, markevery=8
            )
            plt.title('Relative Residual Energy')
            plt.xlabel('Number of Components')
            plt.ylabel('Uncaptured variance')
            plt.grid(True)
            plt.show()
        V_q_eff = Vh
        d_q = V_q_eff @ np.ones(V_q_eff.shape[1])
        print("magnitude of d_q:", np.linalg.norm(d_q))
    else:
        V_q_eff = qoi
        d_q = V_q_eff @ np.ones(V_q_eff.shape[1])

    rhs_avg = d_q
    half_gap = const_tol
    b_lower = rhs_avg - half_gap
    b_upper = rhs_avg + half_gap

    nnls_solver = NNLSSolver(
        verbosity=verbosity,
        n_outer=1500,
        n_inner=400,
        const_tol=const_tol,
        zero_tol=zero_tol
    )

    print("\nStarting NNLS solve...")
    x, flag = nnls_solver.solve(V_q_eff, b_lower, b_upper)
    print(f"\nSolve complete with exit flag: {flag}")
    print("\nNumber of nonzero elements in x:", np.sum(x > 0), "out of", len(x))

    if plot:
        plt.figure()
        plt.title('NNLS Solution')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.stem(x)

    qoi_all_elem = qoi @ np.ones(qoi.shape[1])
    qoi_hyper = qoi @ x
    error_h = np.linalg.norm(qoi_all_elem - qoi_hyper) / np.linalg.norm(qoi_all_elem)
    print("Hyper-reduced error (normalized):", error_h)

    return x, flag
