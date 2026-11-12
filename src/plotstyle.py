"""
One visual system for every figure (validated palette; thin marks; recessive axes; direct labels).

Colour roles
------------
ACTIVE / DRIED   status of a source (blue / red; CVD-validated pair)
WARD             survey units (orange), LANDSCAPE (aqua), CATCHMENT (blue): the three unit families, validated
                 all-pairs; aqua is below 3:1 contrast, so it is always direct-labelled
NULL             random-partition envelopes (grey band), never a hue
SEQ              one-hue blue ramp for magnitudes; DIV blue-grey-red for signed quantities

Usage:  from src.plotstyle import apply, C, panel_label, save
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = dict(
    ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#ffffff",
    active="#2a78d6", dried="#e34948",
    ward="#eb6834", landscape="#1baf7a", catchment="#2a78d6",
    null="#d9d8d2", null_edge="#b9b8b0",
    blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", red="#e34948", violet="#6250d6",
)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])
SEQ_CMAP = SEQ
DIV_CMAP = DIV


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.edgecolor": C["axis"], "axes.linewidth": 0.7, "axes.labelcolor": C["ink2"],
        "axes.titlecolor": C["ink"], "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
        "xtick.color": C["muted"], "ytick.color": C["muted"], "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5, "patch.linewidth": 0,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    })


def panel_label(ax, letter, title=None, x=-0.02, y=1.02):
    """Bold panel letter, optional title in secondary ink."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=C["ink"],
            ha="right", va="bottom")
    if title:
        ax.set_title(title, color=C["ink"], pad=6, loc="left", x=0.0)


def map_axes(ax, aspect_lat=27.57):
    import numpy as np
    ax.set_aspect(1 / np.cos(np.radians(aspect_lat)))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
