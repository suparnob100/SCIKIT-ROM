import numpy as np
import matplotlib.pyplot as plt                 
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, LogLocator

def svd_mode_selector_var(data, tolerance=1e-3, modes=False, **kwargs):
    """
    Select SVD modes based on an uncaptured variance tolerance and plot the uncaptured variance.

    Parameters
    ----------
    data : array_like, shape (n_samples, n_features) or (n_features, n_samples)
        Input data matrix. Columns (or rows) represent snapshots or observations.
    tolerance : float, optional
        Maximum allowed fraction of total variance that remains uncaptured by the selected modes.
        Defaults to 1e-3.
    modes : bool, optional
        If True, prints the number of selected modes. Defaults to False.
    **kwargs
        Additional keyword arguments passed to the plot (e.g., marker style, line width).

    Returns
    -------
    num_selected_modes : int
        Number of SVD modes required to meet the specified uncaptured variance tolerance.
    U : ndarray, shape (n_features, n_features)
        Matrix of left singular vectors from the SVD of the input data.

    Notes
    -----
    - The function computes the full SVD of the (transposed) data matrix and calculates the
      cumulative sum of squared singular values to measure variance content.
    - Uncaptured variance is defined as one minus the cumulative energy.
    - A horizontal line at `y = tolerance` is drawn on the semilog plot for reference.

    Examples
    --------
    >>> num_modes, U = svd_mode_selector_var(data_matrix, tolerance=1e-2)
    >>> print(num_modes)
    5
    """
    data_array = np.asarray(data)
    U, singular_values, _ = np.linalg.svd(data_array.T, full_matrices=False)
    # Fraction of uncaptured variance after k modes
    singular_values_cumsum = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    uncaptured = 1.0 - singular_values_cumsum
    # Avoid zeros for log plotting
    uncaptured[uncaptured < np.finfo(float).eps] = np.finfo(float).eps

    # Determine modes
    idx = np.where(uncaptured < tolerance)[0]
    num_selected_modes = idx[0] + 1 if idx.size > 0 else 1

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(np.arange(1, len(singular_values) + 1), uncaptured, '-o', ms=4, lw=2, **kwargs)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.axhline(y=tolerance, color='black', linestyle='--')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.02)
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Uncaptured variance')

    if modes:
        print(f"Number of modes selected: {num_selected_modes}")

    return num_selected_modes, U


def svd_mode_selector(data, tolerance=1e-3, modes=False, **kwargs):
    """
    Select SVD modes based on relative reconstruction-error tolerance and plot the error.

    Parameters
    ----------
    data : array_like, shape (n_samples, n_features) or (n_features, n_samples)
        Input data matrix. Columns (or rows) represent snapshots or observations.
    tolerance : float, optional
        Maximum allowed relative reconstruction error (L2-norm) for the selected modes.
        Defaults to 1e-3.
    modes : bool, optional
        If True, prints the number of selected modes. Defaults to False.
    **kwargs
        Additional keyword arguments passed to the plot (e.g., marker style, line width).

    Returns
    -------
    num_selected_modes : int
        Number of SVD modes required to meet the specified reconstruction-error tolerance.
    U : ndarray, shape (n_features, n_features)
        Matrix of left singular vectors from the SVD of the input data.

    Notes
    -----
    - Singular values are flipped to compute residual energy from smallest to largest modes.
    - Relative reconstruction error is defined as the square-root of uncaptured energy divided by
      total energy.

    Examples
    --------
    >>> num_modes, U = svd_mode_selector(data_matrix, tolerance=1e-2)
    >>> print(num_modes)
    4
    """
    data_array = np.asarray(data)
    U, singular_values, _ = np.linalg.svd(data_array.T, full_matrices=False)
    # Compute relative reconstruction error
    flipped = singular_values[::-1]
    cum_sq = np.cumsum(flipped**2)
    rel_error = np.sqrt(cum_sq[::-1]) / np.sqrt(cum_sq[-1])

    idx = np.where(rel_error < tolerance)[0]
    num_selected_modes = idx[0] + 1 if idx.size > 0 else 1

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(np.arange(1, len(singular_values) + 1), rel_error, '-o', ms=4, lw=2, **kwargs)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.axhline(y=tolerance, color='black', linestyle='--')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.02)
    ax.set_xlabel('Number of modes')
    ax.set_ylabel('Relative reconstruction error')

    if modes:
        print(f"Number of modes selected: {num_selected_modes}")

    return num_selected_modes, U
