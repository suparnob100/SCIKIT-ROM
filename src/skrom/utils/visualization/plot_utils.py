import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_deim_weights(
    xi: np.ndarray,
    zoom_indices: tuple = (0, 1),
    figsize: tuple = (12, 3),
    nbins: int = 6,
    line_color: str = "#163e64",
    stem_color: str = "#a6a6a6",
    zoom_loc: str = "upper left",
    zoom_size: tuple = ("25%", "40%"),
    zoom_title: str = "Zoomed-in view",
    zoom_box_offset_x: float = -0.5,
    zoom_box_offset_y: float = -0.2,
    zoom_box_width: float = 1.0,
    zoom_box_height_scale: float = 0.4,
    zoom_box_color: str = "red",
    zoom_box_linestyle: str = "--",
    zoom_box_linewidth: float = 1.0
):
    """
    Create a stem plot of `xi` with a zoomed inset.
    Uses constrained_layout for automatic spacing.
    """
    # Use constrained_layout to avoid tight_layout warnings
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Main stem plot
    xs = np.arange(len(xi))
    markerline, stemlines, _ = ax.stem(
        xs, xi, linefmt=line_color, markerfmt="o", basefmt=" "
    )
    markerline.set_color(line_color)
    markerline.set_markersize(5)
    stemlines.set_color(stem_color)

    ax.set_xlabel("Mesh elements")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))

    # Inset zoom
    i0, i1 = zoom_indices
    inset = inset_axes(ax, width=zoom_size[0], height=zoom_size[1], loc=zoom_loc, borderpad=2)
    zoom_x = [i0, i1]
    zoom_y = [xi[i0], xi[i1]]
    m_in, s_in, _ = inset.stem(zoom_x, zoom_y, linefmt=stem_color, markerfmt="o", basefmt=" ")
    m_in.set_color(line_color)
    inset.set_xlim(i0 - 1, i1 + 1)
    inset.set_ylim(0, max(zoom_y) * 1.2)
    inset.set_xticks(zoom_x)
    inset.set_yticks([])
    inset.set_title(zoom_title, fontsize=10)

    # Highlight box
    y0 = xi[i0]
    h = max(zoom_y) * zoom_box_height_scale
    rect = patches.Rectangle(
        (i0 + zoom_box_offset_x, y0 + zoom_box_offset_y),
        zoom_box_width, h,
        edgecolor=zoom_box_color, linestyle=zoom_box_linestyle,
        linewidth=zoom_box_linewidth, fill=False
    )
    ax.add_patch(rect)

    plt.show()
    return fig, ax

def plot_ecsw_weights(
    x_nnls_m: np.ndarray,
    figsize: tuple = (12, 3),
    line_color: str = '#163e64',
    stem_color: str = '#a6a6a6',
    xlabel: str = 'Mesh elements',
    ylabel: str = r'$\xi^*$'
):
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(x_nnls_m))
    markerline, stemlines, _ = ax.stem(
        xs, x_nnls_m,
        linefmt=line_color,
        markerfmt='o',
        basefmt=' '
    )
    markerline.set_color(line_color)
    markerline.set_markersize(5)
    stemlines.set_color(stem_color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.show()
    return fig, ax




def plot_ecsw_weights_3d(
    mesh,
    weights: np.ndarray,
    projection: str = '3d',
    plane: str = 'xy',
    figsize: tuple = (8, 6),
    cmap: str = 'viridis',
    zero_color: str = '#eeeeee',
    zero_alpha: float = 0.3,
    weight_threshold: float = 1e-3
):
    """
    Visualize ECSW weights on a 3D mesh, fading out near-zero weights.

    Parameters
    ----------
    mesh : object
        Mesh with attributes:
         - p: array of shape (3, n_nodes) for node coords.
         - t: array of shape (nodes_per_elem, n_elems).
    weights : 1D ndarray of length n_elems
        ECSW weight per element.
    projection : '3d' or 'projected'
        '3d' = scatter in 3D; 'projected' = heatmap on a plane.
    plane : 'xy', 'xz', or 'yz'
        which plane to project onto if projected.
    figsize : figure size.
    cmap : name of colormap for >threshold weights.
    zero_color : color for weights ≤ threshold.
    zero_alpha : alpha for zero_color points.
    weight_threshold :
        any w ≤ this will be considered “zero” for display.

    Returns
    -------
    fig, ax
    """
    # compute centroids
    centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elems)
    x, y, z = centroids

    # split indices
    zero_idx = weights <= weight_threshold
    nz_idx   = ~zero_idx

    # normalize non-zero weights
    norm = Normalize(vmin=weights[nz_idx].min(),
                     vmax=weights.max())

    if projection == '3d':
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')

        # plot zeros lightly
        ax.scatter(x[zero_idx], y[zero_idx], z[zero_idx],
                   c=zero_color, alpha=zero_alpha, marker='o', s=20)

        # plot non-zeros with colormap
        cmap_obj = cm.get_cmap(cmap)
        ax.scatter(x[nz_idx], y[nz_idx], z[nz_idx],
                   c=weights[nz_idx], cmap=cmap_obj, norm=norm,
                   marker='o', s=30)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj),
                          ax=ax, label='ECSW weight')

    else:
        fig, ax = plt.subplots(figsize=figsize)
        # choose coords
        coord = {'xy':(x,y), 'xz':(x,z), 'yz':(y,z)}[plane]
        ax_coords = coord

        # zeros
        ax.scatter(*(c[zero_idx] for c in ax_coords),
                   c=zero_color, alpha=zero_alpha, marker='o', s=20)
        # non-zeros
        cmap_obj = cm.get_cmap(cmap)
        sc = ax.scatter(*(c[nz_idx] for c in ax_coords),
                        c=weights[nz_idx], cmap=cmap_obj, norm=norm,
                        marker='o', s=30)
        ax.set_xlabel(plane[0].upper()); ax.set_ylabel(plane[1].upper())
        cb = fig.colorbar(sc, ax=ax, label='ECSW weight')

    plt.tight_layout()
    plt.show()
    return fig, ax