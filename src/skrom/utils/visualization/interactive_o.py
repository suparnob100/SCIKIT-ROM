"""Interactive visualization helpers for cached ROM results.

TL;DR
-----
Compare precomputed FOM, ROM, and hyper-ROM fields in notebooks without
calling any solver or regenerating high-fidelity data.
"""

from __future__ import annotations

from collections.abc import Mapping
import re

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


_REFERENCE_NAMES = ("FOM", "HF", "FULL", "REFERENCE")
_REFERENCE_COLOR = "#202124"
_MODEL_COLOR = "#D55E00"
_ERROR_COLOR = "#B2182B"
_GRID_COLOR = "#D9DEE8"
_FONT_CANDIDATES = (
    "Lora",
    "Latin Modern Roman",
    "Latin Modern",
    "STIXGeneral",
    "DejaVu Serif",
)
_CSS_FONT_FAMILY = (
    "Lora, 'Latin Modern Roman', 'Latin Modern', STIXGeneral, "
    "DejaVu Serif, serif"
)
_FONT_FAMILY = next(
    (
        font
        for font in _FONT_CANDIDATES
        if font in {item.name for item in font_manager.fontManager.ttflist}
    ),
    "serif",
)


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
    field_shapes = set()
    for name, values in models.items():
        if values.ndim < 2:
            raise ValueError(
                f"{name!r} must have a leading sample axis and a field axis."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name!r} contains non-finite values.")
        sample_counts.add(values.shape[0])
        field_shapes.add(values.shape[1:])

    if len(sample_counts) != 1:
        raise ValueError("All field arrays must have the same sample count.")
    if len(field_shapes) != 1:
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


def _parameter_labels(labels, parameter_count: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"mu_{index + 1}" for index in range(parameter_count))
    result = tuple(_plain_label(label) for label in labels)
    if len(result) != parameter_count:
        raise ValueError("parameter_labels must match the parameter count.")
    return result


def _plain_label(label) -> str:
    text = str(label).replace("$", "")
    replacements = {
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\delta": "delta",
        r"\kappa": "kappa",
        r"\lambda": "lambda",
        r"\mu": "mu",
        r"\theta": "theta",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


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


def _line_data(field, basis=None, coordinates=None, max_points=5000):
    values = np.asarray(field, dtype=float).reshape(-1)
    if coordinates is not None:
        x = np.asarray(coordinates, dtype=float).reshape(-1)
    elif basis is not None and hasattr(basis, "doflocs"):
        x = np.asarray(basis.doflocs[0], dtype=float).reshape(-1)
    elif basis is not None and hasattr(basis, "mesh"):
        x = np.asarray(basis.mesh.p[0], dtype=float).reshape(-1)
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


def _sample_order(parameters, parameter_index):
    return np.argsort(parameters[:, int(parameter_index)], kind="stable")


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


def _style_axes(ax):
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    show=True,
    y_floor=1.0e-14,
):
    """Plot one cached FOM-vs-ROM snapshot and its pointwise error.

    Parameters
    ----------
    parameters : array_like
        Parameter samples with shape ``(n_samples, n_parameters)``.
    fields : mapping
        Cached fields, for example ``{"FOM": fom, "ROM": rom}``.
    sample_index : int, optional
        Cached sample to display.
    compare : str, optional
        Model name compared against the reference.  If omitted, the first
        non-reference model is used.
    basis, coordinates : optional
        Source for one-dimensional plotting coordinates.
    parameter_labels : sequence of str, optional
        Human-readable parameter names.
    relative_errors, speedups : mapping, optional
        Per-model metrics aligned with the sample axis.
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
    reference_field = _field_at(models[reference], sample_index)
    model_field = _field_at(models[compare], sample_index)
    x_ref, y_ref = _line_data(reference_field, basis, coordinates)
    x_model, y_model = _line_data(model_field, basis, coordinates)
    if x_ref.shape != x_model.shape or not np.allclose(x_ref, x_model):
        raise ValueError("Reference and model coordinates do not align.")

    error = np.abs(y_model - y_ref)
    fig, (ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.1), "hspace": 0.08},
    )
    fig.patch.set_facecolor("white")
    title = _model_title(compare, relative_errors, speedups, sample_index)
    fig.suptitle(
        f"Cached ROM comparison: {title}",
        fontfamily=_FONT_FAMILY,
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.91,
        _parameter_text(parameters, labels, sample_index),
        fontfamily=_FONT_FAMILY,
        fontsize=10,
        color="#40506A",
    )

    ax.plot(
        x_ref,
        y_ref,
        color=_REFERENCE_COLOR,
        linewidth=2.4,
        label=reference,
    )
    ax.plot(
        x_model,
        y_model,
        color=_MODEL_COLOR,
        linestyle="--",
        linewidth=2.0,
        label=compare,
    )
    ax.set_ylabel("solution")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)

    error_ax.semilogy(
        x_ref,
        # np.maximum(error, np.finfo(float).tiny),
        np.maximum(error, y_floor),
        color=_ERROR_COLOR,
        linewidth=1.7,
    )
    error_ax.set_xlabel("x")
    error_ax.set_ylabel("|error|")
    _style_axes(error_ax)
    fig.subplots_adjust(top=0.82, hspace=0.08)
    if show:
        plt.show()
    return fig, (ax, error_ax)


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
    show=True,
):
    """Plot one POD mode as a one-dimensional profile."""
    mode_basis = _pod_basis_array(modes)
    mode_count = mode_basis.shape[1]
    mode_index = int(np.clip(mode_index, 0, mode_count - 1))
    scale = float(scale)
    mode = mode_basis[:, mode_index]

    if add_mean:
        if mean is None:
            raise ValueError("mean is required when add_mean=True.")
        field = np.asarray(mean, dtype=float).reshape(-1) + scale * mode
        ylabel = "mean + scaled mode"
    else:
        field = scale * mode
        ylabel = "scaled mode"

    x, y = _line_data(field, basis=basis, coordinates=coordinates)

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    fig.patch.set_facecolor("white")
    stats = _pod_stats(singular_values, mode_index)
    fig.suptitle(
        f"POD mode {mode_index + 1}{_pod_stat_text(stats)}",
        fontfamily=_FONT_FAMILY,
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    ax.axhline(0.0, color="#AAB2C0", linewidth=0.9, zorder=1)
    if add_mean:
        mean_x, mean_y = _line_data(mean, basis=basis, coordinates=coordinates)
        ax.plot(
            mean_x,
            mean_y,
            color="#7E8798",
            linewidth=1.6,
            label="mean field",
            alpha=0.85,
        )
    ax.plot(
        x,
        y,
        color=_MODEL_COLOR,
        linewidth=2.4,
        label="mean + mode" if add_mean else "mode shape",
    )
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.subplots_adjust(top=0.82)
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

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    fig.patch.set_facecolor("white")
    ax.semilogy(
        modes,
        # np.maximum(error, np.finfo(float).tiny),
        np.maximum(error, y_floor),
        color="#3367D6",
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
        [max(error[selected - 1], np.finfo(float).tiny)],
        color="#009E73",
        s=56,
        zorder=5,
    )
    ax.set_xlabel("retained POD modes")
    ax.set_ylabel("relative truncated reconstruction error")
    ax.set_title(
        "SVD truncation decay",
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


def rom_explorer(
    parameters,
    fields,
    *,
    basis=None,
    coordinates=None,
    parameter_labels=None,
    relative_errors=None,
    speedups=None,
    display_widget=True,
):
    """Create a lightweight cached ROM explorer for notebooks.

    The explorer never calls a solver.  It only switches among arrays already
    present in memory, making it safe to add at the end of existing notebooks.
    """
    models = _model_fields(fields)
    sample_count = next(iter(models.values())).shape[0]
    parameters = _parameter_array(parameters, sample_count)
    labels = _parameter_labels(parameter_labels, parameters.shape[1])
    reference, approximations = _reference_and_models(models)
    if not approximations:
        raise ValueError("fields must include a reference and one model.")

    try:
        import ipywidgets as widgets
        from IPython.display import HTML as DisplayHTML
        from IPython.display import clear_output, display
    except Exception:
        fig, axes = plot_rom_snapshot(
            parameters,
            models,
            sample_index=0,
            compare=approximations[0],
            basis=basis,
            coordinates=coordinates,
            parameter_labels=labels,
            relative_errors=relative_errors,
            speedups=speedups,
            show=display_widget,
        )
        return fig, axes

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

    header = widgets.HTML(
        value=f"""
        <div style="
            border-left: 5px solid #3367d6;
            background: #f1f5ff;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            font-family: {_CSS_FONT_FAMILY};">
          <div style="font-weight: 700; font-size: 18px; color: #17233f;">
            Cached ROM explorer
          </div>
          <div style="font-size: 13px; color: #41516f; margin-top: 4px;">
            FOM comparison with cached arrays only; no solver is called.
          </div>
        </div>
        """
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
                    "<span style='display:inline-block; padding:6px 10px; "
                    "border:1px solid #d6dbe6; border-radius:10px; "
                    "margin-right:6px; margin-bottom:6px; "
                    f"font-family:{_CSS_FONT_FAMILY};'>"
                    f"{label}: <b>{parameters[sample_index, index]:.6g}</b>"
                    "</span>"
                )
                for index, label in enumerate(labels)
            )
            display(
                DisplayHTML(
                    f"<div style='margin: 2px 0 8px 0;'>{badges}</div>"
                )
            )
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
                show=False,
            )
            display(fig)
            plt.close(fig)

    def step(delta):
        sample_slider.value = int(
            np.clip(sample_slider.value + delta, 0, sample_count - 1)
        )

    previous_button.on_click(lambda _: step(-1))
    next_button.on_click(lambda _: step(1))
    sample_slider.observe(render, names="value")
    compare_dropdown.observe(render, names="value")
    order_dropdown.observe(render, names="value")

    controls = widgets.VBox(
        [
            header,
            widgets.HBox(
                [
                    previous_button,
                    next_button,
                    sample_slider,
                    compare_dropdown,
                    order_dropdown,
                ],
                layout=widgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    gap="10px",
                ),
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
    display_widget=True,
):
    """Create a lightweight notebook explorer for POD modes."""
    mode_basis = _pod_basis_array(modes)
    mode_count = mode_basis.shape[1]
    singular_values = _pod_singular_values(singular_values)
    mean = None if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if mean is not None and mean.size != mode_basis.shape[0]:
        raise ValueError("mean must contain one value per spatial DOF.")

    try:
        import ipywidgets as widgets
        from IPython.display import HTML as DisplayHTML
        from IPython.display import clear_output, display
    except Exception:
        fig, ax = plot_pod_mode(
            mode_basis,
            coordinates=coordinates,
            basis=basis,
            singular_values=singular_values,
            mean=mean,
            show=display_widget,
        )
        return fig, ax

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

    header = widgets.HTML(
        value=f"""
        <div style="
            border-left: 5px solid #8E44AD;
            background: #F7F0FF;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            font-family: {_CSS_FONT_FAMILY};">
          <div style="font-weight: 700; font-size: 18px; color: #241436;">
            POD mode explorer
          </div>
          <div style="font-size: 13px; color: #58466F; margin-top: 4px;">
            Inspect one POD basis vector at a time with variance diagnostics.
          </div>
        </div>
        """
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
                        "<div style='margin: 2px 0 8px 0; "
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
                show=False,
            )
            display(fig)
            plt.close(fig)

    def step(delta):
        mode_slider.value = int(
            np.clip(mode_slider.value + delta, 1, mode_count)
        )

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
                layout=widgets.Layout(
                    align_items="center",
                    flex_flow="row wrap",
                    gap="10px",
                ),
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

    try:
        import ipywidgets as widgets
        from IPython.display import HTML as DisplayHTML
        from IPython.display import clear_output, display
    except Exception:
        fig, ax, selected = plot_svd_decay(
            values,
            tolerance=initial_tolerance,
            show=display_widget,
        )
        return fig, ax, selected

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
    header = widgets.HTML(
        value=f"""
        <div style="
            border-left: 5px solid #009E73;
            background: #EEF9F4;
            padding: 12px 16px;
            margin: 4px 0 12px 0;
            font-family: {_CSS_FONT_FAMILY};">
          <div style="font-weight: 700; font-size: 18px; color: #123A2B;">
            SVD truncation explorer
          </div>
          <div style="font-size: 13px; color: #38574B; margin-top: 4px;">
            Move the tolerance to update the selected POD mode count.
          </div>
        </div>
        """
    )

    def render(_=None):
        tolerance = float(tolerance_slider.value)
        selected = _selected_mode_count(error, tolerance)
        with output:
            clear_output(wait=True)
            display(
                DisplayHTML(
                    "<div style='margin: 2px 0 8px 0; "
                    f"font-family:{_CSS_FONT_FAMILY};'>"
                    f"Tolerance: <b>{tolerance:.2e}</b> &nbsp; "
                    f"selected modes: <b>{selected}</b> &nbsp; "
                    f"relative error at selected modes: "
                    f"<b>{error[selected - 1]:.2e}</b>"
                    "</div>"
                )
            )
            fig, _, _ = plot_svd_decay(
                values,
                tolerance=tolerance,
                show=False,
            )
            display(fig)
            plt.close(fig)

    tolerance_slider.observe(render, names="value")
    controls = widgets.VBox([header, tolerance_slider, output])
    if display_widget:
        render()
        display(controls)
    return controls


__all__ = [
    "plot_pod_mode",
    "plot_rom_snapshot",
    "plot_svd_decay",
    "pod_mode_explorer",
    "rom_explorer",
    "svd_decay_explorer",
]
