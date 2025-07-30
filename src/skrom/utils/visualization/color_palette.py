import matplotlib.pyplot as plt
from cycler import cycler

"""
Custom Matplotlib color palettes.

This module provides functions to set and retrieve predefined color palettes
for Matplotlib plots, enabling consistent styling across figures.

[Author: Suparno Bhattacharyya]
"""

def set_color_palette():
    """
    Set a custom color palette for Matplotlib plots.

    Defines and applies a predefined list of hexadecimal color codes to Matplotlib's
    ``axes.prop_cycle``, ensuring a consistent sequence of colors for plot elements.

    Returns
    -------
    colors : list of str
        A list of hexadecimal color strings representing the palette applied.

    Notes
    -----
    - The palette consists of 19 distinct colors chosen for clarity and visual appeal.
    - Applying this palette affects all subsequent plots in the current session.

    Examples
    --------
    >>> colors = set_color_palette()
    >>> plt.plot([0, 1, 2], [10, 20, 15])  # uses the first color in the returned palette
    >>> plt.scatter([1, 2, 3], [5, 10, 20])  # uses the next color in the cycle

    """
    colors = [
        '#1f77b4', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
        '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b',
        '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
        '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
    ]
    plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
    return colors
