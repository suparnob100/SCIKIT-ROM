"""Visualization utility package.

TL;DR
-----
This package groups plotting, color, VTK, and VTU export helpers.

Notes
-----
Use the modules here to prepare ROM results for Matplotlib or ParaView-style visualization.
"""

from .interactive import (
    plot_pod_mode,
    plot_rom_snapshot,
    plot_svd_decay,
    pod_mode_explorer,
    rom_explorer,
    svd_decay_explorer,
)

__all__ = [
    "plot_pod_mode",
    "plot_rom_snapshot",
    "plot_svd_decay",
    "pod_mode_explorer",
    "rom_explorer",
    "svd_decay_explorer",
]
