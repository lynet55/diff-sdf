"""Shared plot style + palette for the bench probes (validated, see dataviz)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRIDLINE = "#e3e2de"
# categorical, fixed order (never cycled)
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
# sequential blue ramp (ordinal use starts at step 250)
SEQ = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": GRIDLINE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
})
