"""Regenerate the four paper figures (PNG 300 dpi, vector PDF, SVG) with matplotlib only.

    python make_figures.py

Data are inlined and copied verbatim from ../figs.py (loop, dose, solve, arena); means are computed
from the per-run values exactly as figs.py does. Style follows the figures4papers conventions
(https://github.com/ChenLiu-1996/figures4papers): Arial/Helvetica sans, top/right spines off,
frameless annotations, blue = method of interest, red = contrast/control, neutral greys = baseline,
no grid clutter, 300 dpi export, editable vector text. Figures are drawn at their final printed
width (7 in, double column), so font sizes below are the printed sizes (7-9 pt).

Encoding shared by figures 2-4 (never colour alone):
    edit            filled blue circle
    control         open red square
    unedited model  filled grey diamond
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

OUT = Path(__file__).resolve().parent

# figures4papers palette (blue = key method, red = contrast, greys = baseline/support)
BLUE = "#0F4D92"
RED = "#B64342"
GREY = "#767676"
DARK = "#272727"
MID = "#4D4D4D"
LIGHT = "#CFCECE"

WIDTH = 7.0  # inches, double column
FS = 8.0  # base font size at printed size
FS_SMALL = 7.0

EDIT = dict(marker="o", mfc=BLUE, mec="white", mew=0.6, ms=5.2, ls="none")
CTRL = dict(marker="s", mfc="white", mec=RED, mew=1.2, ms=4.4, ls="none")
BASE = dict(marker="D", mfc=GREY, mec="white", mew=0.6, ms=4.8, ls="none")
KIND = {"e": (EDIT, BLUE), "c": (CTRL, RED), "n": (BASE, GREY)}


def apply_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": DARK,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.frameon": False,
        "svg.fonttype": "none",  # keep text editable in SVG
        "pdf.fonttype": 42,  # embed TrueType, editable text in PDF
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def finalize_figure(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Figure 1: the loop (schematic)
# ---------------------------------------------------------------------------------------------
LOOP_HEAD = "An LLM author writes the weights of a frozen Gemma 4 E4B as closed-form low-rank additions"
LOOP_STEPS = [("1 Diagnose", "read traces: where is", "the solve rate lost?"),
              ("2 Locate", "find the direction", "in activations"),
              ("3 Write", "closed-form neuron,", "no gradient step"),
              ("4 Predict", "record the expected", "effect first"),
              ("5 Verify", "paired eval, unedited", "+ random control")]
LOOP_BACK = "the outcome, hit or miss, is the next diagnosis"


def fig1_loop():
    W, H = 680.0, 138.0
    fig = plt.figure(figsize=(WIDTH, WIDTH * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # y grows downwards, as in the SVG source
    ax.set_axis_off()

    bw, bh, gap, y, left = 122.0, 62.0, 16.0, 30.0, 3.0
    arrow = dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=MID, lw=1.2,
                 shrinkA=0, shrinkB=0)
    ax.text(left, 14, LOOP_HEAD, fontsize=FS + 0.5, fontweight="bold", va="center")
    for i, (a, b, c) in enumerate(LOOP_STEPS):
        x = left + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=5",
                                    fc="#F4F6FA", ec=BLUE, lw=1.0))
        ax.text(x + 8, y + 15, a, fontsize=FS + 1.5, fontweight="bold", color=BLUE, va="center")
        ax.text(x + 8, y + 34, b, fontsize=FS_SMALL + 0.4, color=DARK, va="center")
        ax.text(x + 8, y + 48, c, fontsize=FS_SMALL + 0.4, color=DARK, va="center")
        if i < len(LOOP_STEPS) - 1:
            ax.annotate("", xy=(x + bw + gap - 1.5, y + bh / 2), xytext=(x + bw + 1.5, y + bh / 2),
                        arrowprops=arrow)
    # dashed return path: verify -> diagnose (iterate)
    x0, x1 = left + bw / 2, left + 4 * (bw + gap) + bw / 2
    yb = y + bh + 20
    ax.plot([x1, x1, x0], [y + bh, yb, yb], color=MID, lw=1.2, ls=(0, (4, 3)),
            solid_capstyle="butt")
    ax.annotate("", xy=(x0, y + bh + 1.5), xytext=(x0, yb),
                arrowprops=dict(arrow, ls=(0, (4, 3))))
    ax.text((x0 + x1) / 2, yb + 13, LOOP_BACK, fontsize=FS, color=MID, ha="center", va="center",
            style="italic")
    finalize_figure(fig, "fig1_loop")


# ---------------------------------------------------------------------------------------------
# Figure 2: dose-response in decision replay
# ---------------------------------------------------------------------------------------------
DOSE_PANELS = [
    dict(title="“Submit now” neuron, wave-5 adapter",
         ylabel="Submit rate at 100\nheld-out decisions",
         xs=[0, .25, .5, 1], ys=[.15, .20, .29, .41], ctl=(.5, .15), ylim=(0, .55), fmt="{:.2f}",
         yticks=[0, .1, .2, .3, .4, .5], tickfmt="{:.1f}",
         ci=[(.093, .233), (.133, .289), (.21, .385), (.319, .508)], ctl_ci=(.093, .233)),
    dict(title="“Think briefly” neuron, base model",
         ylabel="Mean thinking tokens\nper call",
         xs=[0, .25, .5, 1], ys=[659, 594, 524, 399], ctl=(.5, 649), ylim=(300, 700), fmt="{:.0f}",
         yticks=[300, 400, 500, 600, 700], tickfmt="{:.0f}"),
]


def fig2_dose():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    for ax, p, tag in zip(axes, DOSE_PANELS, "ab"):
        xs, ys, (cx, cy), fmt = p["xs"], p["ys"], p["ctl"], p["fmt"]
        ymin, ymax = p["ylim"]
        span = ymax - ymin
        up = ys[-1] > ys[0]
        ax.plot(xs, ys, color=BLUE, lw=2.0, zorder=2, clip_on=False)
        if "ci" in p:  # Wilson 95% intervals (n = 100 decisions)
            for a, b, (lo, hi) in zip(xs, ys, p["ci"]):
                ax.errorbar([a], [b], yerr=[[b - lo], [hi - b]], color=BLUE, lw=1.0, capsize=2.5, zorder=1)
            lo, hi = p["ctl_ci"]
            ax.errorbar([cx + 0.03], [cy], yerr=[[cy - lo], [hi - cy]], color=RED, lw=1.0, capsize=2.5, zorder=1)
        ax.plot(xs, ys, zorder=3, clip_on=False, **dict(EDIT, ms=6.5, mew=1.0))
        for a, b in zip(xs, ys):
            if up:  # label above-left of the rising line
                ax.annotate(fmt.format(b), (a, b), xytext=(-5, 3), textcoords="offset points",
                            ha="right", va="bottom", fontsize=FS)
            elif a == xs[-1]:  # last point of the falling line: label below-left, inside the axes
                ax.annotate(fmt.format(b), (a, b), xytext=(-6, -4), textcoords="offset points",
                            ha="right", va="top", fontsize=FS)
            else:  # label above-right of the falling line
                ax.annotate(fmt.format(b), (a, b), xytext=(5, 5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=FS)
        ax.plot([cx + (0.03 if "ci" in p else 0)], [cy], zorder=4, clip_on=False, **dict(CTRL, ms=5.6, mew=1.5))
        ax.annotate("random-direction\ncontrol " + fmt.format(cy), (cx, cy),
                    xytext=(12 if up else 7, -3 if up else 3), textcoords="offset points", ha="left",
                    va="top" if up else "bottom", fontsize=FS, color=RED, linespacing=1.05)
        # direct label for the edit line
        if up:
            ax.annotate("edit", (xs[-1], ys[-1]), xytext=(-9, -11), textcoords="offset points",
                        ha="left", va="top", fontsize=FS, color=BLUE, fontweight="bold")
        else:
            ax.annotate("edit", (0.75, (ys[2] + ys[3]) / 2), xytext=(5, 5),
                        textcoords="offset points", ha="left", va="bottom", fontsize=FS,
                        color=BLUE, fontweight="bold")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs])
        yt = p["yticks"]
        ax.set_yticks(yt)
        ax.set_yticklabels([p["tickfmt"].format(v) for v in yt])
        ax.set_xlabel("Edit strength (fraction of class gap)")
        ax.set_ylabel(p["ylabel"])
        ax.spines["left"].set_position(("outward", 4))
        ax.spines["bottom"].set_position(("outward", 4))
        ax.text(0.0, 1.07, tag, transform=ax.transAxes, fontsize=FS + 2, fontweight="bold",
                ha="right", va="bottom")
        ax.text(0.04, 1.07, p["title"], transform=ax.transAxes, fontsize=FS + 0.5,
                fontweight="bold", ha="left", va="bottom")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.185, wspace=0.42)
    finalize_figure(fig, "fig2_dose")


# ---------------------------------------------------------------------------------------------
# Figures 3 and 4: one row per arm, one marker per run / batch, thick tick = mean
# ---------------------------------------------------------------------------------------------
SOLVE_ROWS = [
    ("Unedited base", "13 runs", [28, 24, 22, 29, 22, 29, 27, 26, 25, 27, 30, 28, 25], "n"),
    ("Random-direction controls", "14 runs", [27, 23, 26, 27, 27, 25, 28, 24, 23, 27, 25, 25, 25, 26], "c"),
    ("Explore-more 0.125", "4 runs", [31, 30, 28, 33], "e"),
    ("Explore-more 0.25", "7 runs", [32, 29, 30, 31, 29, 26, 27], "e"),
    ("Explore-more 0.5", "10 runs", [28, 31, 33, 30, 26, 32, 30, 24, 28, 27], "e"),
    ("Explore-more 1.0", "1 run", [18], "e"),
]
SOLVE_REF = 26.3  # dashed reference drawn by figs.py, labelled "unedited mean 26.3"

ARENA_ROWS = [
    ("Before-first-edit gate", "Claude, 6 batches", [9, 2, 3, 9, 5, 2], "e"),
    ("Early-calls gate", "Claude, 6 batches", [6, 2, 0, 9, 7, 4], "e"),
    ("Decision-token gate + rotation", "GPT, 6 batches", [4, 0, 2, 8, 2, -2], "e"),
    ("Explore + concise", "Gemini, 3 batches", [7, 3, 7], "e"),
    ("Claude’s control", "4 batches", [1, 1, 4, 1], "c"),
    ("GPT’s control", "4 batches", [-1, -3, 6, 0], "c"),
    ("Gemini’s control", "3 batches", [8, -3, 4], "c"),
]


def _dot_rows(name, rows, xlim, xticks, xtick_fmt, xlabel, ref, ref_label, mean_fmt, height,
              stack=0.235):
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    left, right = 0.285, 0.905
    fig.subplots_adjust(left=left, right=right, top=1 - 0.30 / height, bottom=0.50 / height)
    t_lab = blended_transform_factory(fig.transFigure, ax.transData)

    ax.set_xlim(*xlim)
    ax.set_ylim(len(rows) - 0.5, -0.5)  # first row on top
    for v in xticks:
        ax.axvline(v, color=LIGHT, lw=0.5, zorder=0)
    ax.axvline(ref, color=MID, lw=1.0, ls=(0, (3, 2.5)), zorder=1)
    ax.annotate(ref_label, (ref, -0.5), xytext=(0, 3), textcoords="offset points", ha="center",
                va="bottom", fontsize=FS, color=MID, annotation_clip=False)

    for i, (label, n, vals, kind) in enumerate(rows):
        style, col = KIND[kind]
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F5F5F5", lw=0, zorder=-1)
        m = sum(vals) / len(vals)
        ax.plot([m, m], [i - 0.37, i + 0.37], color=col, lw=2.4, solid_capstyle="butt", zorder=2)
        seen = {}
        for v in vals:  # repeated values are stacked vertically, centred on the row (as figs.py)
            k = seen.get(v, 0)
            seen[v] = k + 1
            ax.plot([v], [i + k * stack - stack / 2 * (vals.count(v) - 1)], zorder=3, **style)
        ax.text(0.012, i - 0.14, label, transform=t_lab, ha="left", va="center", fontsize=FS + 0.5,
                fontweight="bold" if kind == "e" else "normal", color=DARK)
        ax.text(0.012, i + 0.23, n, transform=t_lab, ha="left", va="center", fontsize=FS_SMALL,
                color=MID)
        ax.text(0.988, i, mean_fmt(m), transform=t_lab, ha="right", va="center", fontsize=FS + 0.5,
                fontweight="bold", color=col)
    ax.annotate("mean", (0.988, -0.5), xycoords=t_lab, xytext=(0, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=FS, color=MID, annotation_clip=False)

    ax.set_xticks(xticks)
    ax.set_xticklabels([xtick_fmt(v) for v in xticks])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=4)
    finalize_figure(fig, name)


def fig3_solve():
    _dot_rows("fig3_solve", SOLVE_ROWS, xlim=(16, 36), xticks=list(range(16, 37, 4)),
              xtick_fmt=str,
              xlabel="Tasks solved of 170 (one marker per full evaluation run; bar = mean)",
              ref=SOLVE_REF, ref_label=f"unedited mean {SOLVE_REF}",
              mean_fmt=lambda m: f"{m:.1f}", height=3.0)


def fig4_arena():
    _dot_rows("fig4_arena", ARENA_ROWS, xlim=(-4, 10), xticks=list(range(-4, 11, 2)),
              xtick_fmt=lambda v: f"{v:+d}".replace("-", "−") if v else "0",
              xlabel="Solved tasks minus the unedited arm of the same batch "
                     "(one marker per batch; bar = mean)",
              ref=0, ref_label="unedited", mean_fmt=lambda m: f"{m:+.1f}".replace("-", "−"), height=3.25, stack=0.24)


if __name__ == "__main__":
    apply_publication_style()
    fig1_loop()
    fig2_dose()
    fig3_solve()
    fig4_arena()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("fig*.*"))))
