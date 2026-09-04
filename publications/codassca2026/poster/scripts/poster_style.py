"""Shared style for the CODASSCA 2026 poster figures.

Figures are drawn at their FINAL poster size: figsize is the poster slot in inches
(1 pt = 1/72 in) and font sizes are the point sizes they will have on the A0 sheet.
Body text on the poster is Calibri 24 pt; figure text is 20-26 pt.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]          # publications/codassca2026/poster
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "data"
DPI = 200

# --- fonts: use the same Calibri that PowerPoint will use for the poster text
_DFONTS = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
for f in ("Calibri.ttf", "Calibrib.ttf", "Calibrii.ttf", "Calibriz.ttf", "arial.ttf", "arialbd.ttf"):
    p = _DFONTS / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
FONT = "Calibri" if any("Calibri" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

# --- poster palette (matches the previous Publisher poster + validated chart colors)
INK = "#1a2332"          # navy used for table headers / dark text
INK2 = "#52514e"         # secondary text
GRID = "#e6e6e3"
SAGE = "#c4d0b5"; SAGE_LIGHT = "#d8e0ce"; GOLD = "#e2d5a3"
# selectors (fixed order everywhere): proposed, supervised MI, PCA, variance, random
C_OURS = "#c0392b"; C_MI = "#2a78d6"; C_PCA = "#4a3aa7"; C_VAR = "#eda100"; C_RAND = "#7f8c8d"
C_FULL = "#0b0b0b"       # full-channel ceiling reference line (ink, dashed)
METHOD_STYLE = {   # name -> (color, marker, linestyle)
    "Proposed (label-free)": (C_OURS, "o", "-"),
    "Supervised MI":         (C_MI, "s", "-"),
    "PCA":                   (C_PCA, "v", "-"),
    "Variance":              (C_VAR, "^", "-"),
    "Random-K":              (C_RAND, "D", "--"),
}
# latent-factor clusters / sensor groups (fixed order), noise is neutral
C_CLUSTERS = ["#1baf7a", "#eb6834", "#e87ba4", "#4a3aa7"]
C_NOISE = "#cfcfcb"
SEQ_CMAP = "Blues"       # one-hue sequential ramp for relevance / influence
EEM_CMAP = "YlOrRd"      # neighbour-hue warm ramp for fluorescence intensity

# --- type scale on the A0 sheet (points)
FS_TITLE = 26; FS_LABEL = 22; FS_TICK = 20; FS_LEGEND = 20; FS_ANNOT = 20; FS_SMALL = 18

plt.rcParams.update({
    "font.family": FONT, "font.size": FS_LABEL,
    "axes.titlesize": FS_TITLE, "axes.titleweight": "bold", "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
    "axes.edgecolor": "#8a8a86", "axes.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "text.color": INK, "axes.titlecolor": INK,
    "lines.linewidth": 3.0, "lines.markersize": 11, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def slot(width_pt: float, height_pt: float):
    """Figure sized to a poster slot given in points."""
    return plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))


def save(fig, name: str, dpi: int = DPI, pad: float = 0.02):
    out = FIG / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print("wrote", out)
    return out


def despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
