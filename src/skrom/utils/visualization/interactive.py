"""Interactive visualization helpers for cached ROM results.

TL;DR
-----
Compare precomputed FOM, ROM, and hyper-ROM fields in notebooks without
calling any solver or regenerating high-fidelity data.

This module supports four common visualization cases:

* 1D scalar fields as line plots;
* 2D scalar FEM fields as triangular color plots;
* 3D scalar FEM fields as colored nodal scatter plots;
* 2D/3D vector displacement fields as deformed wireframes.

The widget helpers are intentionally lightweight.  They depend only on
Matplotlib and ipywidgets; if ipywidgets is unavailable, the same functions
fall back to static Matplotlib figures.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import base64
import re
import warnings

from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
import numpy as np
from matplotlib.ticker import MaxNLocator

try:  # only needed for 3D wireframe plots
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except Exception:  # pragma: no cover
    Line3DCollection = None


_REFERENCE_NAMES = ("FOM", "HF", "FULL", "REFERENCE")

# Matplotlib plot colors are intentionally left unchanged.  The theme system
# below affects only the notebook widget panels and the selected font.
_REFERENCE_COLOR = "#202124"
_MODEL_COLOR = "#D55E00"
_MODEL_COLOR_ALT = "#3367D6"
_ERROR_COLOR = "#B2182B"
_MESH_COLOR = "#AAB2C0"

_INTERACTIVE_THEME = "pubassist"
_PREFERRED_FONT = None

_PANEL_THEMES = {
    "pubassist": {
        "panel": "#FFFDF7",
        "panel_alt": "#F3EEE2",
        "border": "#C9BA98",
        "text": "#4A3D35",
        "muted": "#7C7165",
        "heading": "#15100D",
        "accent": "#81283B",
        "accent_dark": "#5C1A29",
    },
    "minimal": {
        "panel": "#FFFFFF",
        "panel_alt": "#F7F8FA",
        "border": "#D8DEE8",
        "text": "#2F3742",
        "muted": "#667085",
        "heading": "#111827",
        "accent": "#4B5563",
        "accent_dark": "#1F2937",
    },
}


def _panel_theme():
    return _PANEL_THEMES.get(_INTERACTIVE_THEME, _PANEL_THEMES["minimal"])


def interactive_themes():
    """Return available notebook panel themes."""
    return tuple(_PANEL_THEMES)


def _default_font_dirs():
    candidates = []
    try:
        here = Path(__file__).resolve().parent
        candidates.extend(
            (
                here / "assets" / "fonts",
                here.parent / "assets" / "fonts",
                here.parent.parent / "assets" / "fonts",
            )
        )
    except Exception:
        pass
    try:
        candidates.append(Path.cwd() / "fonts")
    except Exception:
        pass
    try:
        candidates.append(Path.home() / ".local" / "share" / "fonts" / "skrom_fonts")
    except Exception:
        pass

    result = []
    seen = set()
    for candidate in candidates:
        try:
            path = Path(candidate).expanduser().resolve()
        except Exception:
            continue
        if path in seen or not path.exists():
            continue
        seen.add(path)
        result.append(path)
    return tuple(result)


def register_interactive_fonts(*font_dirs):
    """Register local/package fonts with Matplotlib for this Python session."""
    search_dirs = tuple(Path(path) for path in font_dirs) if font_dirs else _default_font_dirs()
    registered = []
    for font_dir in search_dirs:
        if not font_dir.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for font_file in font_dir.rglob(pattern):
                try:
                    font_manager.fontManager.addfont(str(font_file))
                    registered.append(str(font_file))
                except Exception as exc:
                    warnings.warn(f"Could not register font {font_file}: {exc}", stacklevel=2)
    return tuple(registered)


def _available_font_names():
    return {item.name for item in font_manager.fontManager.ttflist}


def _select_font(preferred=None):
    candidates = tuple(
        font
        for font in (
            preferred,
            _PREFERRED_FONT,
            "Lora",
            "Latin Modern Roman",
            "Latin Modern",
            "Roboto Slab",
            "Roboto",
            "Georgia",
            "STIXGeneral",
            "DejaVu Serif",
        )
        if font
    )
    available = _available_font_names()
    return next((font for font in candidates if font in available), "DejaVu Serif")


def _set_font_globals(preferred=None):
    global _FONT_FAMILY, _CSS_FONT_FAMILY
    _FONT_FAMILY = _select_font(preferred)
    _CSS_FONT_FAMILY = (
        f"'{_FONT_FAMILY}', Lora, 'Latin Modern Roman', 'Latin Modern', "
        "Georgia, 'DejaVu Serif', serif"
    )


def _apply_font_rcparams():
    # Deliberately change fonts only. Plot colors, backgrounds, colormaps, and
    # other Matplotlib styling are controlled by the user's own style file.
    plt.rcParams.update(
        {
            "font.family": _FONT_FAMILY,
            "font.serif": [
                _FONT_FAMILY,
                "Lora",
                "Latin Modern Roman",
                "Latin Modern",
                "Georgia",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
        }
    )


def set_interactive_theme(name="pubassist", *, font=None):
    """Set the notebook panel theme. Matplotlib plot colors are not changed."""
    global _INTERACTIVE_THEME, _PREFERRED_FONT
    normalized = str(name).strip().lower()
    if normalized not in _PANEL_THEMES:
        raise ValueError(
            f"Unknown theme {name!r}. Available themes: {interactive_themes()}"
        )
    _INTERACTIVE_THEME = normalized
    if font is not None:
        _PREFERRED_FONT = str(font)
    register_interactive_fonts()
    _set_font_globals(font)
    _apply_font_rcparams()
    return _panel_theme().copy()


def set_interactive_font(font="Roboto"):
    """Set the preferred font used by widget panels and Matplotlib text."""
    global _PREFERRED_FONT
    _PREFERRED_FONT = None if font is None else str(font)
    register_interactive_fonts()
    _set_font_globals(_PREFERRED_FONT)
    _apply_font_rcparams()
    return _FONT_FAMILY


def interactive_font_status():
    """Return selected and available interactive fonts for debugging."""
    return {
        "selected_font": _FONT_FAMILY,
        "preferred_font": _PREFERRED_FONT,
        "available_matches": sorted(
            name
            for name in _available_font_names()
            if any(key in name for key in ("Lora", "Latin Modern", "Roboto"))
        ),
        "searched_font_dirs": tuple(str(path) for path in _default_font_dirs()),
    }


@lru_cache(maxsize=8)
def _font_face_css(font_family):
    try:
        path = Path(font_manager.findfont(font_family, fallback_to_default=False))
    except Exception:
        return ""
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".ttf":
        mime = "font/ttf"
        fmt = "truetype"
    elif suffix == ".otf":
        mime = "font/otf"
        fmt = "opentype"
    else:
        return ""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return (
        "@font-face {"
        f"font-family: '{font_family}';"
        f"src: url(data:{mime};base64,{encoded}) format('{fmt}');"
        "font-weight: 400 800; font-style: normal; font-display: swap;"
        "}"
    )


def _widget_css():
    theme = _panel_theme()
    return f"""
    <style>
      {_font_face_css(_FONT_FAMILY)}
      .skrom-panel, .skrom-panel * {{
        font-family: {_CSS_FONT_FAMILY} !important;
      }}
      .skrom-panel {{
        border: 1px solid {theme['border']};
        border-left: 5px solid {theme['accent']};
        background: {theme['panel_alt']};
        padding: 12px 16px;
        margin: 4px 0 12px 0;
      }}
      .skrom-panel-title {{
        font-weight: 700;
        font-size: 18px;
        color: {theme['heading']};
        letter-spacing: 0.1px;
      }}
      .skrom-panel-subtitle {{
        font-size: 13px;
        color: {theme['muted']};
        margin-top: 4px;
        line-height: 1.45;
      }}
      .skrom-badge {{
        display: inline-block;
        padding: 6px 10px;
        border: 1px solid {theme['border']};
        border-radius: 8px;
        background: {theme['panel']};
        color: {theme['text']};
        margin-right: 6px;
        margin-bottom: 6px;
        font-family: {_CSS_FONT_FAMILY} !important;
      }}
    </style>
    """


register_interactive_fonts()
_set_font_globals()
_apply_font_rcparams()


# ---------------------------------------------------------------------------
# Basic validation and formatting
# ---------------------------------------------------------------------------


def _model_fields(fields) -> dict[str, np.ndarray]:
    if isinstance(fields, Mapping):
        models = {
            str(name): np.asarray(values, dtype=float)
            for name, values in fields.items()
        }
    else:
        models = {"Field": np.asarray(fields, dtype=float)}
    if not models:
        raise ValueError("fields must contain at least one model.")

    sample_counts = set()
    trailing_shapes = set()
    for name, values in models.items():
        if values.ndim < 2:
            raise ValueError(
                f"{name!r} must have a leading sample axis and a field axis."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name!r} contains non-finite values.")
        sample_counts.add(values.shape[0])
        trailing_shapes.add(values.shape[1:])

    if len(sample_counts) != 1:
        raise ValueError("All field arrays must have the same sample count.")
    if len(trailing_shapes) != 1:
        raise ValueError("All field arrays must have matching field shapes.")
    return models


def _parameter_array(parameters, sample_count: int) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] != sample_count:
        raise ValueError(
            "parameters must have shape (n_samples, n_parameters); "
            f"expected {sample_count} samples, got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters contains non-finite values.")
    return values


def _plain_label(label) -> str:
    text = str(label).replace("$", "")
    replacements = {
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\delta": "delta",
        r"\epsilon": "epsilon",
        r"\eta": "eta",
        r"\kappa": "kappa",
        r"\lambda": "lambda",
        r"\mu": "mu",
        r"\nu": "nu",
        r"\theta": "theta",
        r"\rho": "rho",
        r"\sigma": "sigma",
        r"\phi": "phi",
        r"\omega": "omega",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _parameter_labels(labels, parameter_count: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"mu_{index + 1}" for index in range(parameter_count))
    result = tuple(_plain_label(label) for label in labels)
    if len(result) != parameter_count:
        raise ValueError("parameter_labels must match the parameter count.")
    return result


def _reference_and_models(models):
    names = list(models)
    reference = next(
        (
            name
            for candidate in _REFERENCE_NAMES
            for name in names
            if name.upper() == candidate
        ),
        names[0],
    )
    approximations = [name for name in names if name != reference]
    return reference, approximations


def _metric_value(metrics, model, sample_index):
    if metrics is None or model not in metrics:
        return None
    values = np.asarray(metrics[model], dtype=float).reshape(-1)
    if not 0 <= sample_index < values.size:
        return None
    return float(values[sample_index])


def _field_at(values: np.ndarray, sample_index: int, time_index=None):
    field = np.asarray(values[sample_index], dtype=float)
    if time_index is not None:
        field = field[int(time_index)]
    return field.reshape(-1)


def _parameter_text(parameters, labels, sample_index):
    entries = [
        f"{label}: {value:.6g}"
        for label, value in zip(labels, parameters[sample_index])
    ]
    return " | ".join(entries)


def _model_title(model, relative_errors, speedups, sample_index):
    parts = [model]
    error = _metric_value(relative_errors, model, sample_index)
    speedup = _metric_value(speedups, model, sample_index)
    if error is not None:
        parts.append(f"error {error:.3g}%")
    if speedup is not None:
        parts.append(f"speed-up {speedup:.3g}x")
    return " | ".join(parts)


# def _style_axes(ax, *, hide_top_right=True):
#     ax.grid(False)
#     if hide_top_right:
#         try:
#             ax.spines["top"].set_visible(False)
#             ax.spines["right"].set_visible(False)
#         except Exception:
#             pass

def _style_axes(ax, *, hide_top_right=True, max_ticks=5):
    theme = _panel_theme()

    ax.grid(False)

    # Limit tick crowding without changing plot colors or plot data.
    # Keep log-scale ticks unchanged for semilogy plots.
    try:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        if ax.get_yscale() == "linear":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
        if hasattr(ax, "zaxis"):
            ax.zaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    except Exception:
        pass

    # try:
    #     ax.set_facecolor(theme["panel"])
    # except Exception:
    #     pass

    # try:
    # ax.spines["left"].set_color(theme["border"])
    # ax.spines["bottom"].set_color(theme["border"])
    # ax.spines["left"].set_linewidth(0.9)
    # ax.spines["bottom"].set_linewidth(0.9)
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["top"].set_color(theme["border"])
        ax.spines["right"].set_color(theme["border"])
    # except Exception:
    #     pass

    # try:
    #     ax.tick_params(colors=theme["muted"], labelsize=9)
    #     ax.xaxis.label.set_color(theme["text"])
    #     ax.yaxis.label.set_color(theme["text"])
    #     if hasattr(ax, "zaxis"):
    #         ax.zaxis.label.set_color(theme["text"])
    # except Exception:
    #     pass

    title = ax.get_title()
    if title:
        ax.set_title(
            title,
            fontfamily=_FONT_FAMILY,
            fontweight="bold",
            color=theme["heading"],
        )
# ---------------------------------------------------------------------------
# FEM geometry helpers
# ---------------------------------------------------------------------------


def _basis_dimension(basis) -> int | None:
    if basis is None or not hasattr(basis, "mesh"):
        return None
    return int(np.asarray(basis.mesh.p).shape[0])


def _node_count(basis) -> int | None:
    if basis is None or not hasattr(basis, "mesh"):
        return None
    return int(np.asarray(basis.mesh.p).shape[1])


def _nodal_dofs(basis):
    if basis is None or not hasattr(basis, "nodal_dofs"):
        return None
    return np.asarray(basis.nodal_dofs)


def _is_vector_basis(basis) -> bool:
    nodal_dofs = _nodal_dofs(basis)
    return nodal_dofs is not None and nodal_dofs.ndim == 2 and nodal_dofs.shape[0] > 1


def _is_scalar_basis(basis) -> bool:
    if basis is None or not hasattr(basis, "mesh"):
        return False
    nodal_dofs = _nodal_dofs(basis)
    if nodal_dofs is None:
        return True
    return nodal_dofs.ndim == 2 and nodal_dofs.shape[0] == 1


def _coordinate_array(coordinates):
    if coordinates is None:
        return None
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim == 1:
        return coords[np.newaxis, :]
    if coords.ndim != 2:
        return None
    # prefer shape (dim, n_points); allow (n_points, dim)
    if coords.shape[0] <= 3:
        return coords
    if coords.shape[1] <= 3:
        return coords.T
    return coords


def _render_kind(kind: str, field: np.ndarray, basis=None, coordinates=None) -> str:
    """Choose how a field should be displayed.

    Returns one of ``line``, ``scalar2d``, ``scalar3d``, or ``displacement``.
    """
    normalized = str(kind).strip().lower()
    allowed = {"auto", "line", "scalar", "scalar2d", "scalar3d", "displacement"}
    if normalized not in allowed:
        raise ValueError(
            'kind must be "auto", "line", "scalar", "scalar2d", '
            '"scalar3d", or "displacement".'
        )
    if normalized == "scalar":
        dimension = _basis_dimension(basis)
        return "scalar3d" if dimension == 3 else "scalar2d"
    if normalized != "auto":
        return normalized

    values = np.asarray(field).reshape(-1)
    coords = _coordinate_array(coordinates)
    if coords is not None:
        if coords.shape[0] == 1:
            return "line"
        if coords.shape[0] == 2 and coords.shape[1] == values.size:
            return "scalar2d"
        if coords.shape[0] == 3 and coords.shape[1] == values.size:
            return "scalar3d"

    dimension = _basis_dimension(basis)
    if basis is not None and _is_vector_basis(basis):
        return "displacement"
    if basis is not None and _is_scalar_basis(basis):
        if dimension == 1:
            return "line"
        if dimension == 2:
            return "scalar2d"
        if dimension == 3:
            return "scalar3d"
    return "line"


def _line_data(field: np.ndarray, basis=None, coordinates=None, max_points=5000):
    values = np.asarray(field, dtype=float).reshape(-1)
    coords = _coordinate_array(coordinates)
    if coords is not None:
        x = coords[0].reshape(-1)
    elif basis is not None and hasattr(basis, "mesh"):
        points = np.asarray(basis.mesh.p, dtype=float)
        x = points[0].reshape(-1)
        nodal_dofs = _nodal_dofs(basis)
        if (
            nodal_dofs is not None
            and values.size == getattr(basis, "N", -1)
            and nodal_dofs.ndim == 2
            and nodal_dofs.shape[0] == 1
        ):
            values = values[nodal_dofs[0]]
    else:
        x = np.arange(values.size, dtype=float)

    if x.size != values.size:
        raise ValueError(
            f"field has {values.size} values but coordinates has {x.size}."
        )

    order = np.argsort(x, kind="stable")
    if order.size > max_points:
        keep = np.linspace(0, order.size - 1, max_points, dtype=np.intp)
        order = order[keep]
    return x[order], values[order]


def _nodal_scalar(field, basis=None, coordinates=None):
    values = np.asarray(field, dtype=float).reshape(-1)
    coords = _coordinate_array(coordinates)
    if coords is not None:
        if coords.shape[1] != values.size:
            raise ValueError(
                f"coordinates contains {coords.shape[1]} points but field has {values.size}."
            )
        return values

    if basis is None or not hasattr(basis, "mesh"):
        return values
    node_count = _node_count(basis)
    if values.size == node_count:
        return values

    nodal_dofs = _nodal_dofs(basis)
    if (
        nodal_dofs is not None
        and values.size == getattr(basis, "N", -1)
        and nodal_dofs.ndim == 2
        and nodal_dofs.shape[0] == 1
    ):
        return values[nodal_dofs[0]]

    raise ValueError(
        f"Scalar field has {values.size} values; expected {node_count} "
        f"nodal values or {getattr(basis, 'N', 'basis.N')} basis values."
    )


def _nodal_displacement(field, basis):
    values = np.asarray(field, dtype=float).reshape(-1)
    if basis is None:
        raise ValueError("basis is required for displacement plotting.")
    if values.size != getattr(basis, "N", -1):
        raise ValueError(
            f"Displacement field has {values.size} values; expected {basis.N}."
        )
    nodal_dofs = _nodal_dofs(basis)
    if nodal_dofs is None or nodal_dofs.ndim != 2:
        raise ValueError("basis.nodal_dofs is required for displacement plotting.")
    dimension = _basis_dimension(basis)
    if nodal_dofs.shape[0] < dimension:
        raise ValueError(
            "The vector basis does not provide one nodal component per mesh dimension."
        )
    return values[nodal_dofs[:dimension]]


def _triangles(mesh):
    connectivity = np.asarray(mesh.t, dtype=np.intp)
    points = np.asarray(mesh.p)
    if points.shape[0] != 2:
        raise ValueError("Triangle plotting requires a 2D mesh.")
    if connectivity.shape[0] == 3:
        return connectivity.T
    if connectivity.shape[0] == 4:
        first = connectivity[[0, 1, 2]].T
        second = connectivity[[0, 2, 3]].T
        return np.vstack((first, second))
    raise ValueError("2D scalar plotting requires a triangle or quadrilateral mesh.")


def _mesh_edges(mesh):
    points = np.asarray(mesh.p)
    dimension = points.shape[0]
    if dimension == 1:
        connectivity = np.asarray(mesh.t, dtype=np.intp)
        return connectivity[:2].T
    if dimension == 2 and hasattr(mesh, "facets"):
        return np.asarray(mesh.facets, dtype=np.intp).T
    if dimension == 3 and hasattr(mesh, "edges"):
        if hasattr(mesh, "boundary_edges"):
            edge_indices = np.asarray(mesh.boundary_edges(), dtype=np.intp)
            return np.asarray(mesh.edges, dtype=np.intp)[:, edge_indices].T
        return np.asarray(mesh.edges, dtype=np.intp).T
    if dimension == 3 and hasattr(mesh, "facets"):
        facets = np.asarray(mesh.facets, dtype=np.intp)
        if hasattr(mesh, "boundary_facets"):
            facets = facets[:, np.asarray(mesh.boundary_facets(), dtype=np.intp)]
        collected = set()
        for face in facets.T:
            for left, right in combinations(np.asarray(face).tolist(), 2):
                collected.add(tuple(sorted((int(left), int(right)))))
        return np.asarray(sorted(collected), dtype=np.intp)

    connectivity = np.asarray(mesh.t, dtype=np.intp)
    collected = {
        tuple(sorted((int(left), int(right))))
        for element in connectivity.T
        for left, right in combinations(element.tolist(), 2)
    }
    return np.asarray(sorted(collected), dtype=np.intp)


def _segments(points, edges):
    points = np.asarray(points, dtype=float)
    return points.T[np.asarray(edges, dtype=np.intp)]


def _scalar_color_limits(values_list, symmetric=False):
    values = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values_list])
    if symmetric:
        limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        return -limit, limit
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(minimum, maximum):
        padding = max(abs(minimum), 1.0) * 1.0e-12
        minimum -= padding
        maximum += padding
    return minimum, maximum


def _pod_displacement_scale(mode, basis):
    displacement = _nodal_displacement(mode, basis)
    maximum = float(np.linalg.norm(displacement, axis=0).max())
    if maximum <= np.finfo(float).eps:
        return 1.0
    spans = np.ptp(np.asarray(basis.mesh.p, dtype=float), axis=1)
    positive = spans[spans > np.finfo(float).eps]
    length = float(positive.min()) if positive.size else float(np.linalg.norm(spans))
    if length <= np.finfo(float).eps:
        length = 1.0
    return 0.15 * length / maximum


# ---------------------------------------------------------------------------
# Static plotting: ROM snapshots
# ---------------------------------------------------------------------------


def _plot_line_snapshot(
    fig,
    reference_field,
    model_field,
    reference_name,
    model_name,
    basis,
    coordinates,
    y_floor,
):
    grid = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.1), hspace=0.08)
    ax = fig.add_subplot(grid[0])
    error_ax = fig.add_subplot(grid[1], sharex=ax)
    ax.tick_params(labelbottom=False)

    x_ref, y_ref = _line_data(reference_field, basis, coordinates)
    x_model, y_model = _line_data(model_field, basis, coordinates)
    if x_ref.shape != x_model.shape or not np.allclose(x_ref, x_model):
        raise ValueError("Reference and model coordinates do not align.")

    ax.plot(x_ref, y_ref, color=_REFERENCE_COLOR, linewidth=2.4, label=reference_name)
    ax.plot(x_model, y_model, color=_MODEL_COLOR, linestyle="--", linewidth=2.0, label=model_name)
    ax.set_ylabel("solution")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)

    error = np.abs(y_model - y_ref)
    error_ax.semilogy(x_ref, np.maximum(error, y_floor), color=_ERROR_COLOR, linewidth=1.7)
    error_ax.set_xlabel("$x$")
    error_ax.set_ylabel("error")
    _style_axes(error_ax)
    return np.asarray([ax, error_ax], dtype=object)


def _plot_scalar2d_snapshot(
    fig,
    reference_field,
    model_field,
    reference_name,
    model_name,
    basis,
    coordinates,
):
    if coordinates is not None:
        coords = _coordinate_array(coordinates)
        if coords is None or coords.shape[0] < 2:
            raise ValueError("2D scalar plotting requires 2D coordinates.")
        x, y = coords[0], coords[1]
        triangles = None
    else:
        points = np.asarray(basis.mesh.p, dtype=float)
        x, y = points[0], points[1]
        triangles = _triangles(basis.mesh)

    ref = _nodal_scalar(reference_field, basis, coordinates)
    mod = _nodal_scalar(model_field, basis, coordinates)
    err = np.abs(mod - ref)
    vmin, vmax = _scalar_color_limits([ref, mod])
    emax = max(float(err.max()), np.finfo(float).eps)

    axes = np.empty(3, dtype=object)
    for index, (name, values, cmap, limits) in enumerate(
        (
            (reference_name, ref, "viridis", (vmin, vmax)),
            (model_name, mod, "viridis", (vmin, vmax)),
            (f"|{model_name} - {reference_name}|", err, "magma", (0.0, emax)),
        )
    ):
        ax = fig.add_subplot(1, 3, index + 1)
        axes[index] = ax
        triangulation = None if triangles is None else Triangulation(x, y, triangles)
        if triangulation is None:
            im = ax.tricontourf(x, y, values, levels=40, cmap=cmap, vmin=limits[0], vmax=limits[1])
        else:
            im = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(name, fontfamily=_FONT_FAMILY, fontweight="bold")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal", adjustable="box")
        _style_axes(ax)
        label = "absolute error" if index == 2 else "solution"
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, label=label)
    return axes


def _plot_scalar3d_snapshot(
    fig,
    reference_field,
    model_field,
    reference_name,
    model_name,
    basis,
    coordinates,
):
    if coordinates is not None:
        coords = _coordinate_array(coordinates)
        if coords is None or coords.shape[0] < 3:
            raise ValueError("3D scalar plotting requires 3D coordinates.")
        points = coords[:3]
    else:
        points = np.asarray(basis.mesh.p, dtype=float)
        if points.shape[0] != 3:
            raise ValueError("3D scalar plotting requires a 3D mesh.")

    ref = _nodal_scalar(reference_field, basis, coordinates)
    mod = _nodal_scalar(model_field, basis, coordinates)
    err = np.abs(mod - ref)
    vmin, vmax = _scalar_color_limits([ref, mod])
    emax = max(float(err.max()), np.finfo(float).eps)

    axes = np.empty(3, dtype=object)
    for index, (name, values, cmap, limits) in enumerate(
        (
            (reference_name, ref, "viridis", (vmin, vmax)),
            (model_name, mod, "viridis", (vmin, vmax)),
            (f"|{model_name} - {reference_name}|", err, "magma", (0.0, emax)),
        )
    ):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        axes[index] = ax
        im = ax.scatter(
            points[0],
            points[1],
            points[2],
            c=values,
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
            s=8,
            depthshade=False,
        )
        ax.set_title(name, fontfamily=_FONT_FAMILY, fontweight="bold")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        try:
            ax.set_box_aspect(np.maximum(np.ptp(points, axis=1), 1e-9))
        except Exception:
            pass
        label = "absolute error" if index == 2 else "solution"
        fig.colorbar(im, ax=ax, shrink=0.70, pad=0.04, label=label)
    return axes


def _add_wire(ax, points, edges, *, color, linewidth, label=None, alpha=1.0):
    points = np.asarray(points, dtype=float)
    edges = np.asarray(edges, dtype=np.intp)
    dimension = points.shape[0]
    if dimension == 2:
        collection = LineCollection(
            _segments(points, edges),
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            label=label,
        )
        ax.add_collection(collection)
    elif dimension == 3:
        if Line3DCollection is None:
            raise RuntimeError("Line3DCollection could not be imported.")
        collection = Line3DCollection(
            _segments(points, edges),
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            label=label,
        )
        ax.add_collection3d(collection)
    else:
        raise ValueError("Wireframe plotting supports only 2D and 3D meshes.")
    return collection


def _set_equal_limits(ax, points):
    points = np.asarray(points, dtype=float)
    mins = points.min(axis=1)
    maxs = points.max(axis=1)
    spans = np.maximum(maxs - mins, 1.0e-12)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * float(spans.max())
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    if points.shape[0] >= 2:
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
    if points.shape[0] == 3:
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
    elif points.shape[0] == 2:
        ax.set_aspect("equal", adjustable="box")


def _plot_displacement_snapshot(
    fig,
    reference_field,
    model_field,
    reference_name,
    model_name,
    basis,
    deformation_scale,
):
    points = np.asarray(basis.mesh.p, dtype=float)
    edges = _mesh_edges(basis.mesh)
    dimension = points.shape[0]
    projection = "3d" if dimension == 3 else None

    ax = fig.add_subplot(1, 2, 1, projection=projection)
    error_ax = fig.add_subplot(1, 2, 2, projection=projection)

    ref_u = _nodal_displacement(reference_field, basis)
    mod_u = _nodal_displacement(model_field, basis)
    scale = float(deformation_scale)
    ref_points = points + scale * ref_u
    mod_points = points + scale * mod_u
    combined = np.hstack((points, ref_points, mod_points))

    _add_wire(ax, points, edges, color=_MESH_COLOR, linewidth=0.7, alpha=0.6, label="undeformed")
    _add_wire(ax, ref_points, edges, color=_REFERENCE_COLOR, linewidth=1.6, label=reference_name)
    _add_wire(ax, mod_points, edges, color=_MODEL_COLOR, linewidth=1.2, label=model_name)
    ax.set_title(f"{reference_name} and {model_name}", fontfamily=_FONT_FAMILY, fontweight="bold")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if dimension == 3:
        ax.set_zlabel("$z$")
    ax.legend(loc="best", fontsize="small")
    _set_equal_limits(ax, combined)
    _style_axes(ax, hide_top_right=dimension != 3)

    err = np.linalg.norm(mod_u - ref_u, axis=0)
    if dimension == 2:
        scatter = error_ax.scatter(ref_points[0], ref_points[1], c=err, cmap="magma", s=14)
        error_ax.set_xlabel("$x$")
        error_ax.set_ylabel("$y$")
    else:
        scatter = error_ax.scatter(ref_points[0], ref_points[1], ref_points[2], c=err, cmap="magma", s=12, depthshade=False)
        error_ax.set_xlabel("$x$")
        error_ax.set_ylabel("$y$")
        error_ax.set_zlabel("$z$")
    error_ax.set_title("nodal displacement error", fontfamily=_FONT_FAMILY, fontweight="bold")
    _set_equal_limits(error_ax, combined)
    _style_axes(error_ax, hide_top_right=dimension != 3)
    fig.colorbar(scatter, ax=error_ax, shrink=0.75, pad=0.04, label="|u error|")
    return np.asarray([ax, error_ax], dtype=object)


def plot_rom_snapshot(
    parameters,
    fields,
    *,
    sample_index=0,
    compare=None,
    basis=None,
    coordinates=None,
    parameter_labels=None,
    relative_errors=None,
    speedups=None,
    time_index=None,
    kind="auto",
    deformation_scale=1.0,
    show=True,
    y_floor=1.0e-14,
    figsize=None,
    fig=None,
):
    """Plot one cached FOM-vs-ROM snapshot and its error.

    Parameters
    ----------
    parameters : array_like
        Parameter samples with shape ``(n_samples, n_parameters)``.
    fields : mapping
        Cached fields, for example ``{"FOM": fom, "ROM": rom}``.
    sample_index : int, optional
        Cached sample to display.
    compare : str, optional
        Model name compared against the reference. If omitted, the first
        non-reference model is used.
    basis, coordinates : optional
        FEM basis or explicit coordinates for plotting.
    kind : {"auto", "line", "scalar", "scalar2d", "scalar3d", "displacement"}
        Rendering mode. ``auto`` is usually sufficient.
    deformation_scale : float, optional
        Scale factor for displacement plots.
    show : bool, optional
        If True, call ``plt.show()``.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    models = _model_fields(fields)
    sample_count = next(iter(models.values())).shape[0]
    parameters = _parameter_array(parameters, sample_count)
    labels = _parameter_labels(parameter_labels, parameters.shape[1])
    reference, approximations = _reference_and_models(models)
    if compare is None:
        if not approximations:
            raise ValueError("At least one non-reference model is required.")
        compare = approximations[0]
    if compare not in models or compare == reference:
        raise ValueError(f"compare must be one of {approximations}.")

    sample_index = int(np.clip(sample_index, 0, sample_count - 1))
    reference_field = _field_at(models[reference], sample_index, time_index)
    model_field = _field_at(models[compare], sample_index, time_index)
    selected_kind = _render_kind(kind, reference_field, basis, coordinates)

    title = _model_title(compare, relative_errors, speedups, sample_index)
    if figsize is None:
        if selected_kind == "line":
            figsize = (8.0, 5.4)
        elif selected_kind in {"scalar2d", "scalar3d"}:
            figsize = (13.0, 4.4)
        else:
            figsize = (12.5, 5.5)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.clear()
        if figsize is not None:
            fig.set_size_inches(*figsize)
    fig.patch.set_facecolor("white")

    if selected_kind == "line":
        axes = _plot_line_snapshot(
            fig,
            reference_field,
            model_field,
            reference,
            compare,
            basis,
            coordinates,
            y_floor,
        )
    elif selected_kind == "scalar2d":
        axes = _plot_scalar2d_snapshot(
            fig,
            reference_field,
            model_field,
            reference,
            compare,
            basis,
            coordinates,
        )
    elif selected_kind == "scalar3d":
        axes = _plot_scalar3d_snapshot(
            fig,
            reference_field,
            model_field,
            reference,
            compare,
            basis,
            coordinates,
        )
    else:
        axes = _plot_displacement_snapshot(
            fig,
            reference_field,
            model_field,
            reference,
            compare,
            basis,
            deformation_scale,
        )

    fig.suptitle(
        f"Cached ROM comparison: {title}",
        fontfamily=_FONT_FAMILY,
        fontsize=14,
        fontweight="bold",
        x=0.02,
        y=0.985,
        ha="left",
    )

    fig.text(
        0.02,
        0.915,
        _parameter_text(parameters, labels, sample_index),
        fontfamily=_FONT_FAMILY,
        fontsize=10,
        color="#40506A",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))

    if show:
        plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# POD and SVD utilities
# ---------------------------------------------------------------------------


def _pod_basis_array(modes):
    values = np.asarray(modes, dtype=float)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("modes must be a non-empty 2D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("modes contains non-finite values.")
    return values


def _pod_singular_values(singular_values):
    if singular_values is None:
        return None
    values = np.asarray(singular_values, dtype=float).reshape(-1)
    if values.size == 0:
        return None
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must contain finite non-negative values.")
    return values


def _relative_truncation_error(singular_values):
    values = _pod_singular_values(singular_values)
    if values is None:
        raise ValueError("singular_values must be supplied.")
    total = float(np.sum(values * values))
    if total <= 0.0:
        return np.zeros_like(values)
    trailing = np.cumsum(values[::-1] ** 2)[::-1]
    error_after_k = np.concatenate((trailing[1:], np.array([0.0])))
    return np.sqrt(error_after_k / total)


def _selected_mode_count(relative_error, tolerance):
    values = np.asarray(relative_error, dtype=float).reshape(-1)
    selected = np.flatnonzero(values <= float(tolerance))
    return int(selected[0] + 1) if selected.size else values.size


def _pod_stats(singular_values, mode_index):
    values = _pod_singular_values(singular_values)
    if values is None or mode_index >= values.size:
        return None
    total = float(np.sum(values * values))
    if total <= 0.0:
        return None
    variance = float(values[mode_index] ** 2 / total)
    cumulative = float(np.sum(values[: mode_index + 1] ** 2) / total)
    ratio = float(values[mode_index] / values[0]) if values[0] > 0.0 else 0.0
    return ratio, variance, cumulative


def _pod_stat_text(stats):
    if stats is None:
        return ""
    ratio, variance, cumulative = stats
    return (
        f" | singular value ratio {ratio:.3e}"
        f" | variance {100.0 * variance:.3f}%"
        f" | cumulative {100.0 * cumulative:.3f}%"
    )


def _plot_scalar2d_mode(fig, field, basis, coordinates, title):
    if coordinates is not None:
        coords = _coordinate_array(coordinates)
        x, y = coords[0], coords[1]
        triangles = None
    else:
        points = np.asarray(basis.mesh.p, dtype=float)
        x, y = points[0], points[1]
        triangles = _triangles(basis.mesh)
    values = _nodal_scalar(field, basis, coordinates)
    limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    ax = fig.add_subplot(1, 1, 1)
    triangulation = None if triangles is None else Triangulation(x, y, triangles)
    if triangulation is None:
        im = ax.tricontourf(x, y, values, levels=40, cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        im = ax.tripcolor(triangulation, values, shading="gouraud", cmap="coolwarm", vmin=-limit, vmax=limit)
    fig.colorbar(im, ax=ax, label="mode amplitude")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontfamily=_FONT_FAMILY, fontsize=13, fontweight="bold")
    _style_axes(ax)
    return ax


def _plot_scalar3d_mode(fig, field, basis, coordinates, title):
    if coordinates is not None:
        coords = _coordinate_array(coordinates)
        points = coords[:3]
    else:
        points = np.asarray(basis.mesh.p, dtype=float)
    values = _nodal_scalar(field, basis, coordinates)
    limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    im = ax.scatter(points[0], points[1], points[2], c=values, cmap="coolwarm", vmin=-limit, vmax=limit, s=10, depthshade=False)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.04, label="mode amplitude")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(title, fontfamily=_FONT_FAMILY, fontsize=13, fontweight="bold")
    try:
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=1), 1e-9))
    except Exception:
        pass
    return ax


def _plot_displacement_mode(fig, mode, basis, title, *, mean=None, scale=1.0, add_mean=False):
    points = np.asarray(basis.mesh.p, dtype=float)
    edges = _mesh_edges(basis.mesh)
    dimension = points.shape[0]
    projection = "3d" if dimension == 3 else None
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    mode_scale = _pod_displacement_scale(mode, basis) * float(scale)
    mode_u = _nodal_displacement(mode, basis)
    if add_mean:
        if mean is None:
            raise ValueError("mean is required when add_mean=True.")
        reference_u = _nodal_displacement(mean, basis)
    else:
        reference_u = np.zeros_like(mode_u)
    deformed = points + reference_u + mode_scale * mode_u

    _add_wire(ax, points, edges, color=_MESH_COLOR, linewidth=0.6, alpha=0.55, label="reference")
    _add_wire(ax, deformed, edges, color=_MODEL_COLOR, linewidth=1.4, label="mode shape")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if dimension == 3:
        ax.set_zlabel("$z$")
    ax.set_title(title, fontfamily=_FONT_FAMILY, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize="small")
    _set_equal_limits(ax, np.hstack((points, deformed)))
    _style_axes(ax, hide_top_right=dimension != 3)
    return ax


def plot_pod_mode(
    modes,
    *,
    mode_index=0,
    coordinates=None,
    basis=None,
    singular_values=None,
    mean=None,
    scale=1.0,
    add_mean=False,
    kind="auto",
    show=True,
):
    """Plot one POD mode.

    The plotting style is selected automatically: 1D scalar modes are line
    plots, 2D scalar modes are colored fields, 3D scalar modes are nodal
    scatter plots, and vector-valued modes are deformed wireframes.
    """
    mode_basis = _pod_basis_array(modes)
    mode_count = mode_basis.shape[1]
    mode_index = int(np.clip(mode_index, 0, mode_count - 1))
    mode = mode_basis[:, mode_index]

    mean_array = None if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if mean_array is not None and mean_array.size != mode.size:
        raise ValueError("mean must contain one value per spatial DOF.")

    selected_kind = _render_kind(kind, mode, basis, coordinates)
    stats = _pod_stats(singular_values, mode_index)
    # title = f"POD mode {mode_index + 1}{_pod_stat_text(stats)}"
    title = ""

    if selected_kind == "line":
        if add_mean:
            if mean_array is None:
                raise ValueError("mean is required when add_mean=True.")
            field = mean_array + float(scale) * mode
            ylabel = "mean + scaled mode"
        else:
            field = float(scale) * mode
            ylabel = "scaled mode"
        x, y = _line_data(field, basis=basis, coordinates=coordinates)
        fig, ax = plt.subplots(figsize=(0.85 * 8.0, 0.85 * 4.3))
        fig.patch.set_facecolor("white")
        # ax.axhline(0.0, color="#AAB2C0", linewidth=0.9, zorder=1)
        if add_mean and mean_array is not None:
            mean_x, mean_y = _line_data(mean_array, basis=basis, coordinates=coordinates)
            ax.plot(mean_x, mean_y, color="#7E8798", linewidth=1.6, label="mean field", alpha=0.85)
        ax.plot(x, y, color=_MODEL_COLOR, linewidth=2.4, label="mean + mode" if add_mean else "mode shape")
        ax.set_xlabel("$x$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=True)
        ax.set_title(title, fontfamily=_FONT_FAMILY, fontsize=13, fontweight="bold", loc="left")
        _style_axes(ax)
    elif selected_kind == "scalar2d":
        field = float(scale) * mode
        if add_mean:
            if mean_array is None:
                raise ValueError("mean is required when add_mean=True.")
            field = mean_array + field
        fig = plt.figure(figsize=(0.7*6.5, 0.7*5.2))
        fig.patch.set_facecolor("white")
        ax = _plot_scalar2d_mode(fig, field, basis, coordinates, title)
    elif selected_kind == "scalar3d":
        field = float(scale) * mode
        if add_mean:
            if mean_array is None:
                raise ValueError("mean is required when add_mean=True.")
            field = mean_array + field
        fig = plt.figure(figsize=(0.7*7.0, 0.7*5.8))
        fig.patch.set_facecolor("white")
        ax = _plot_scalar3d_mode(fig, field, basis, coordinates, title)
    else:
        fig = plt.figure(figsize=(0.7 * 8.0, 0.7 * 5.6))
        fig.patch.set_facecolor("white")
        ax = _plot_displacement_mode(
            fig,
            mode,
            basis,
            title,
            mean=mean_array,
            scale=scale,
            add_mean=add_mean,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_svd_decay(
    singular_values,
    *,
    tolerance=1.0e-4,
    y_floor=1.0e-14,
    show=True,
):
    """Plot relative truncated reconstruction error versus retained modes."""
    values = _pod_singular_values(singular_values)
    error = _relative_truncation_error(values)
    modes = np.arange(1, values.size + 1)
    selected = _selected_mode_count(error, tolerance)

    fig, ax = plt.subplots(figsize=(0.87 * 7.6, 0.87 * 4.3))
    fig.patch.set_facecolor("white")
    ax.semilogy(
        modes,
        np.maximum(error, y_floor),
        color=_MODEL_COLOR_ALT,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="truncated reconstruction error",
    )
    ax.axhline(
        tolerance,
        color=_ERROR_COLOR,
        linestyle="--",
        linewidth=1.8,
        label=f"tolerance = {tolerance:.1e}",
    )
    ax.axvline(
        selected,
        color="#009E73",
        linestyle="-.",
        linewidth=1.8,
        label=f"selected modes = {selected}",
    )
    ax.scatter(
        [selected],
        [max(error[selected - 1], y_floor)],
        color="#009E73",
        s=56,
        zorder=5,
    )
    ax.set_xlabel("retained POD modes")
    ax.set_ylabel("relative truncated reconstruction error")
    ax.set_title(
        " ",
        fontfamily=_FONT_FAMILY,
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(loc="best", frameon=True)
    ax.set_xlim(1, max(1, values.size))
    ax.set_ylim(y_floor, 1.0)
    _style_axes(ax)
    fig.subplots_adjust(top=0.88)
    if show:
        plt.show()
    return fig, ax, selected


# ---------------------------------------------------------------------------
# Widget explorers
# ---------------------------------------------------------------------------


def _import_widgets():
    try:
        import ipywidgets as widgets
        from IPython.display import HTML as DisplayHTML
        from IPython.display import clear_output, display
        return widgets, DisplayHTML, clear_output, display
    except Exception:
        return None


def _widget_header(title, subtitle, *, color=None, background=None, text=None):
    imported = _import_widgets()
    if imported is None:
        return None
    widgets, _, _, _ = imported
    return widgets.HTML(
        value=f"""
        {_widget_css()}
        <div class="skrom-panel">
          <div class="skrom-panel-title">{title}</div>
          <div class="skrom-panel-subtitle">{subtitle}</div>
        </div>
        """
    )


def _sample_order(parameters, parameter_index):
    return np.argsort(parameters[:, int(parameter_index)], kind="stable")


def rom_explorer(
    parameters,
    fields,
    *,
    basis=None,
    coordinates=None,
    parameter_labels=None,
    relative_errors=None,
    speedups=None,
    kind="auto",
    deformation_scale=1.0,
    display_widget=True,
):
    """Create a lightweight cached ROM explorer for notebooks.

    The explorer never calls a solver. It only switches among arrays already
    present in memory, making it safe to add at the end of existing notebooks.
    """
    models = _model_fields(fields)
    sample_count = next(iter(models.values())).shape[0]
    parameters = _parameter_array(parameters, sample_count)
    labels = _parameter_labels(parameter_labels, parameters.shape[1])
    reference, approximations = _reference_and_models(models)
    if not approximations:
        raise ValueError("fields must include a reference and one model.")

    imported = _import_widgets()
    if imported is None:
        return plot_rom_snapshot(
            parameters,
            models,
            sample_index=0,
            compare=approximations[0],
            basis=basis,
            coordinates=coordinates,
            parameter_labels=labels,
            relative_errors=relative_errors,
            speedups=speedups,
            kind=kind,
            deformation_scale=deformation_scale,
            show=display_widget,
        )

    widgets, DisplayHTML, clear_output, display = imported
    order_dropdown = widgets.Dropdown(
        options=[(label, index) for index, label in enumerate(labels)],
        value=0,
        description="Order by",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="210px"),
    )
    compare_dropdown = widgets.Dropdown(
        options=approximations,
        value=approximations[0],
        description="Compare",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="210px"),
    )
    sample_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=sample_count - 1,
        step=1,
        description="Sample",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )
    previous_button = widgets.Button(
        description="Previous",
        icon="chevron-left",
        layout=widgets.Layout(width="120px"),
    )
    next_button = widgets.Button(
        description="Next",
        icon="chevron-right",
        layout=widgets.Layout(width="120px"),
    )
    output = widgets.Output()
    header = _widget_header(
        "Cached ROM explorer",
        "FOM comparison with cached arrays only; no solver is called.",
        color="#3367d6",
        background="#f1f5ff",
        text="#17233f",
    )

    def current_index():
        order = _sample_order(parameters, order_dropdown.value)
        return int(order[sample_slider.value])

    def render(_=None):
        with output:
            clear_output(wait=True)
            sample_index = current_index()
            badges = " ".join(
                (
                    "<span class='skrom-badge'>"
                    f"{label}: <b>{parameters[sample_index, index]:.6g}</b>"
                    "</span>"
                )
                for index, label in enumerate(labels)
            )
            display(DisplayHTML(f"{_widget_css()}<div style='margin: 2px 0 8px 0;'>{badges}</div>"))
            fig, _ = plot_rom_snapshot(
                parameters,
                models,
                sample_index=sample_index,
                compare=compare_dropdown.value,
                basis=basis,
                coordinates=coordinates,
                parameter_labels=labels,
                relative_errors=relative_errors,
                speedups=speedups,
                kind=kind,
                deformation_scale=deformation_scale,
                show=False,
            )
            display(fig)
            plt.close(fig)

    def step(delta):
        sample_slider.value = int(np.clip(sample_slider.value + delta, 0, sample_count - 1))

    previous_button.on_click(lambda _: step(-1))
    next_button.on_click(lambda _: step(1))
    sample_slider.observe(render, names="value")
    compare_dropdown.observe(render, names="value")
    order_dropdown.observe(render, names="value")

    controls = widgets.VBox(
        [
            header,
            widgets.HBox(
                [previous_button, next_button, sample_slider, compare_dropdown, order_dropdown],
                layout=widgets.Layout(align_items="center", flex_flow="row wrap", gap="10px"),
            ),
            output,
        ]
    )
    if display_widget:
        render()
        display(controls)
    return controls


def pod_mode_explorer(
    modes,
    *,
    coordinates=None,
    basis=None,
    singular_values=None,
    mean=None,
    kind="auto",
    display_widget=True,
):
    """Create a notebook explorer for POD modes.

    The explorer automatically switches between line, 2D scalar, 3D scalar,
    and displacement visualizations.
    """
    mode_basis = _pod_basis_array(modes)
    mode_count = mode_basis.shape[1]
    singular_values = _pod_singular_values(singular_values)
    mean = None if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if mean is not None and mean.size != mode_basis.shape[0]:
        raise ValueError("mean must contain one value per spatial DOF.")

    imported = _import_widgets()
    if imported is None:
        return plot_pod_mode(
            mode_basis,
            coordinates=coordinates,
            basis=basis,
            singular_values=singular_values,
            mean=mean,
            kind=kind,
            show=display_widget,
        )

    widgets, DisplayHTML, clear_output, display = imported
    mode_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=mode_count,
        step=1,
        description="Mode",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="420px"),
    )
    previous_button = widgets.Button(
        description="Previous",
        icon="chevron-left",
        layout=widgets.Layout(width="120px"),
    )
    next_button = widgets.Button(
        description="Next",
        icon="chevron-right",
        layout=widgets.Layout(width="120px"),
    )
    scale_slider = widgets.FloatSlider(
        value=1.0,
        min=0.05,
        max=5.0,
        step=0.05,
        description="Scale",
        continuous_update=False,
        readout_format=".2f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="300px"),
    )
    add_mean_box = widgets.Checkbox(
        value=False,
        description="Add mean field",
        disabled=mean is None,
        indent=False,
        layout=widgets.Layout(width="150px"),
    )
    output = widgets.Output()
    header = _widget_header(
        "POD mode explorer",
        "Inspect one POD basis vector at a time with variance diagnostics.",
        color="#8E44AD",
        background="#F7F0FF",
        text="#241436",
    )

    def render(_=None):
        mode_index = mode_slider.value - 1
        stats = _pod_stats(singular_values, mode_index)
        with output:
            clear_output(wait=True)
            if stats is not None:
                ratio, variance, cumulative = stats
                display(
                    DisplayHTML(
                        f"{_widget_css()}<div style='margin: 2px 0 8px 0; "
                        f"font-family:{_CSS_FONT_FAMILY};'>"
                        f"<b>Mode {mode_index + 1}</b> &nbsp; "
                        f"SV ratio: <b>{ratio:.3e}</b> &nbsp; "
                        f"variance: <b>{100.0 * variance:.3f}%</b> &nbsp; "
                        f"cumulative: <b>{100.0 * cumulative:.3f}%</b>"
                        "</div>"
                    )
                )
            fig, _ = plot_pod_mode(
                mode_basis,
                mode_index=mode_index,
                coordinates=coordinates,
                basis=basis,
                singular_values=singular_values,
                mean=mean,
                scale=scale_slider.value,
                add_mean=add_mean_box.value,
                kind=kind,
                show=False,
            )
            display(fig)
            plt.close(fig)

    def step(delta):
        mode_slider.value = int(np.clip(mode_slider.value + delta, 1, mode_count))

    previous_button.on_click(lambda _: step(-1))
    next_button.on_click(lambda _: step(1))
    mode_slider.observe(render, names="value")
    scale_slider.observe(render, names="value")
    add_mean_box.observe(render, names="value")

    controls = widgets.VBox(
        [
            header,
            widgets.HBox(
                [previous_button, next_button, mode_slider, scale_slider, add_mean_box],
                layout=widgets.Layout(align_items="center", flex_flow="row wrap", gap="10px"),
            ),
            output,
        ]
    )
    if display_widget:
        render()
        display(controls)
    return controls


def svd_decay_explorer(
    singular_values,
    *,
    initial_tolerance=1.0e-4,
    display_widget=True,
):
    """Create an interactive SVD truncation tolerance explorer."""
    values = _pod_singular_values(singular_values)
    error = _relative_truncation_error(values)
    positive_error = error[error > 0.0]
    min_exp = -14 if positive_error.size == 0 else int(
        np.floor(np.log10(max(float(positive_error.min()) * 0.1, 1.0e-14)))
    )
    max_exp = 0
    initial_tolerance = float(initial_tolerance)
    initial_tolerance = min(max(initial_tolerance, 10.0**min_exp), 1.0)

    imported = _import_widgets()
    if imported is None:
        return plot_svd_decay(values, tolerance=initial_tolerance, show=display_widget)

    widgets, DisplayHTML, clear_output, display = imported
    tolerance_slider = widgets.FloatLogSlider(
        value=initial_tolerance,
        base=10,
        min=min_exp,
        max=max_exp,
        step=0.05,
        description="Tolerance",
        continuous_update=False,
        readout_format=".1e",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="520px"),
    )
    output = widgets.Output()
    header = _widget_header(
        "SVD truncation explorer",
        "Move the tolerance to update the selected POD mode count.",
        color="#009E73",
        background="#EEF9F4",
        text="#123A2B",
    )

    def render(_=None):
        tolerance = float(tolerance_slider.value)
        selected = _selected_mode_count(error, tolerance)
        with output:
            clear_output(wait=True)
            display(
                DisplayHTML(
                    f"{_widget_css()}<div style='margin: 2px 0 8px 0; "
                    f"font-family:{_CSS_FONT_FAMILY};'>"
                    f"Tolerance: <b>{tolerance:.2e}</b> &nbsp; "
                    f"selected modes: <b>{selected}</b> &nbsp; "
                    f"relative error at selected modes: "
                    f"<b>{error[selected - 1]:.2e}</b>"
                    "</div>"
                )
            )
            fig, _, _ = plot_svd_decay(values, tolerance=tolerance, show=False)
            display(fig)
            plt.close(fig)

    tolerance_slider.observe(render, names="value")
    controls = widgets.VBox([header, tolerance_slider, output])
    if display_widget:
        render()
        display(controls)
    return controls


# ---------------------------------------------------------------------------
# Convenience wrappers and animation helpers
# ---------------------------------------------------------------------------


def rom_explorer_from_problem(problem, *, display_widget=True, **kwargs):
    """Build a cached ROM explorer from a problem object when possible."""
    fields = {"FOM": np.asarray(problem.fos_test_data)}
    errors = {}
    speedups = {}

    rom_solutions = np.asarray(getattr(problem, "rom_solutions", []))
    if rom_solutions.size:
        fields["ROM"] = rom_solutions
        rom_errors = np.asarray(getattr(problem, "rom_error", []))
        rom_speedups = np.asarray(getattr(problem, "speed_up", []))
        if rom_errors.size:
            errors["ROM"] = rom_errors
        if rom_speedups.size:
            speedups["ROM"] = rom_speedups

    for name, result in getattr(problem, "rom_results", {}).items():
        solutions = np.asarray(result.get("solutions", []))
        if solutions.size:
            fields[str(name)] = solutions
        if result.get("relative_errors") is not None:
            errors[str(name)] = np.asarray(result["relative_errors"])
        if result.get("speedups") is not None:
            speedups[str(name)] = np.asarray(result["speedups"])

    sample_count = min(values.shape[0] for values in fields.values())
    fields = {name: values[:sample_count] for name, values in fields.items()}
    errors = {
        name: values[:sample_count]
        for name, values in errors.items()
        if values.size >= sample_count
    }
    speedups = {
        name: values[:sample_count]
        for name, values in speedups.items()
        if values.size >= sample_count
    }
    parameters = np.asarray(problem.param_list_test)[:sample_count]

    return rom_explorer(
        parameters,
        fields,
        basis=getattr(problem, "basis", None),
        relative_errors=errors or None,
        speedups=speedups or None,
        display_widget=display_widget,
        **kwargs,
    )


def animate_parameter_sweep(
    parameters,
    fields,
    *,
    basis=None,
    coordinates=None,
    parameter_labels=None,
    parameter_index=0,
    compare=None,
    kind="auto",
    deformation_scale=1.0,
    relative_errors=None,
    speedups=None,
    interval=350,
    repeat=True,
    figsize=None,
):
    """Animate cached FOM/ROM fields over a parameter sweep."""
    from matplotlib.animation import FuncAnimation

    models = _model_fields(fields)
    sample_count = next(iter(models.values())).shape[0]
    params = _parameter_array(parameters, sample_count)
    labels = _parameter_labels(parameter_labels, params.shape[1])
    order = _sample_order(params, int(parameter_index))
    reference, approximations = _reference_and_models(models)
    if compare is None:
        compare = approximations[0]

    first_field = _field_at(models[reference], int(order[0]))
    selected_kind = _render_kind(kind, first_field, basis, coordinates)
    if figsize is None:
        if selected_kind == "line":
            figsize = (8.0, 5.4)
        elif selected_kind in {"scalar2d", "scalar3d"}:
            figsize = (13.0, 4.4)
        else:
            figsize = (12.5, 5.5)
    fig = plt.figure(figsize=figsize)

    def update(frame):
        fig.clear()
        sample_index = int(order[frame])
        plot_rom_snapshot(
            params,
            models,
            sample_index=sample_index,
            compare=compare,
            basis=basis,
            coordinates=coordinates,
            parameter_labels=labels,
            relative_errors=relative_errors,
            speedups=speedups,
            kind=selected_kind,
            deformation_scale=deformation_scale,
            figsize=figsize,
            fig=fig,
            show=False,
        )
        return tuple(fig.axes)

    animation = FuncAnimation(
        fig,
        update,
        frames=order.size,
        interval=float(interval),
        repeat=bool(repeat),
        blit=False,
    )
    animation._skrom_frame_order = order
    return animation


def display_animation(animation, *, format="jshtml"):
    """Display a Matplotlib animation inside a notebook."""
    from IPython.display import HTML, display

    selected = str(format).strip().lower()
    if selected == "jshtml":
        html = animation.to_jshtml()
    elif selected in {"html5", "video", "html5_video"}:
        html = animation.to_html5_video()
    else:
        raise ValueError('format must be "jshtml" or "html5".')
    result = HTML(html)
    display(result)
    return result


def save_animation(animation, output, *, fps=60, dpi=120):
    """Save a parameter animation as a GIF or MP4."""
    from pathlib import Path

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        writer = "pillow"
    elif suffix == ".mp4":
        writer = "ffmpeg"
    else:
        raise ValueError("Animation output must end in .gif or .mp4.")
    animation.save(path, writer=writer, fps=float(fps), dpi=int(dpi))
    return path


__all__ = [
    "animate_parameter_sweep",
    "display_animation",
    "interactive_font_status",
    "interactive_themes",
    "plot_pod_mode",
    "plot_rom_snapshot",
    "plot_svd_decay",
    "pod_mode_explorer",
    "rom_explorer",
    "rom_explorer_from_problem",
    "save_animation",
    "set_interactive_font",
    "set_interactive_theme",
    "svd_decay_explorer",
]
