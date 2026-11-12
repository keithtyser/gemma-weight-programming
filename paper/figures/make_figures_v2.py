"""Revised paper figures (v2): every figure carries a one-line takeaway title, direct labels, and the
comparison the reader needs (edit vs matched control vs unedited). Data are inlined from the paper's
tables and result files (sources noted per figure). Writes PNG (300 dpi), PDF and SVG next to this file.

    python make_figures_v2.py
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt

from make_figures import (OUT, BLUE, RED, GREY, DARK, MID, LIGHT, WIDTH, FS, FS_SMALL, EDIT, CTRL, BASE,
                          KIND, apply_publication_style, finalize_figure)

GREEN = "#2E7D4F"
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}  # two-sided 95% t quantiles, df = n - 1


def title(fig, text, y=0.985, x=0.012):
    fig.text(x, y, text, ha="left", va="top", fontsize=FS + 1.5, fontweight="bold", color=DARK)


# ---------------------------------------------------------------------------------------------
# Figure 1: the loop (top) and the edit it writes (bottom)
# ---------------------------------------------------------------------------------------------
STEPS = [("1  Diagnose", "where is solving lost?", "e.g. 8+ calls before", "the first edit"),
         ("2  Locate", "class-mean direction", "for that decision;", "check held-out AUC"),
         ("3  Write", "closed-form edit to one", "MLP neuron per layer", "+ matched control"),
         ("4  Predict", "behaviour change and", "solve-rate interval,", "before any run"),
         ("5  Verify", "replay held-out", "decisions; episodes vs", "unedited, same batch"),
         ("6  Score", "outcome inside the", "interval? It feeds", "the next diagnosis")]


def fig1_loop():
    W, H = 700.0, 330.0
    fig = plt.figure(figsize=(WIDTH, WIDTH * H / W))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    ax.text(4, 12, "a   The loop: an LLM author edits a frozen agent's weights without gradients",
            fontsize=FS + 1.5, fontweight="bold", va="center")
    bw, bh, gap, y0, x0 = 103.0, 64.0, 13.4, 28.0, 3.0
    arrow = dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=MID, lw=1.1, shrinkA=0, shrinkB=0)
    for i, (h, a, b, c) in enumerate(STEPS):
        x = x0 + i * (bw + gap)
        gpu = i == 4
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=5",
                                    fc="#FDF3E6" if gpu else "#F4F6FA", ec="#B8660B" if gpu else BLUE, lw=1.0))
        ax.text(x + 7, y0 + 13, h, fontsize=FS + 0.8, fontweight="bold", color="#B8660B" if gpu else BLUE, va="center")
        for k, s in enumerate((a, b, c)):
            ax.text(x + 5, y0 + 29 + 11.5 * k, s, fontsize=FS_SMALL - 0.6, color=DARK, va="center")
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(x + bw + gap - 1, y0 + bh / 2), xytext=(x + bw + 1, y0 + bh / 2), arrowprops=arrow)
    xa, xb, yb = x0 + bw / 2, x0 + 5 * (bw + gap) + bw / 2, y0 + bh + 14
    ax.plot([xb, xb, xa], [y0 + bh, yb, yb], color=MID, lw=1.1, ls=(0, (4, 3)))
    ax.annotate("", xy=(xa, y0 + bh + 1), xytext=(xa, yb), arrowprops=dict(arrow, ls=(0, (4, 3))))
    ax.text((xa + xb) / 2, yb + 10, "iterate", fontsize=FS_SMALL, color=MID, ha="center", style="italic")
    ax.text(x0 + 4 * (bw + gap) + bw / 2, y0 - 4, "only step 5 uses a GPU", fontsize=FS_SMALL - 0.4, color=RED, ha="center", va="bottom")

    # ---- panel b: the edit itself
    yt = 140
    ax.text(4, yt, "b   What step 3 writes: one quiet MLP neuron per layer, stored as a LoRA adapter",
            fontsize=FS + 1.5, fontweight="bold", va="center")
    yy = yt + 8
    def box(x, y, w, h, txt, ec=DARK, fc="white", fs=FS_SMALL, bold=False, color=DARK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", linespacing=1.25)
    # residual stream line
    ax.annotate("", xy=(W - 8, yy + 132), xytext=(8, yy + 132), arrowprops=dict(arrow, color=GREY, lw=1.6))
    ax.text(10, yy + 144, "residual stream $x$ at layers 18, 20, 22", fontsize=FS_SMALL, color=MID)
    box(20, yy + 22, 92, 40, "RMS norm\n" + r"$\rightarrow\ \hat{x}$", fc="#F7F7F7")
    ax.annotate("", xy=(66, yy + 62), xytext=(66, yy + 128), arrowprops=dict(arrow, color=GREY))
    # neuron box
    box(140, yy + 4, 250, 98, "", ec=BLUE, fc="#F4F6FA")
    ax.text(150, yy + 17, "the overwritten neuron  j", fontsize=FS_SMALL + 0.5, fontweight="bold", color=BLUE)
    ax.text(150, yy + 37, r"gate$_j = G\,q$   ($G = 4$;  $q\cdot\hat{x} \approx 1$ everywhere)", fontsize=FS_SMALL, color=DARK)
    ax.text(150, yy + 53, r"   or  gate$_j \propto r - \theta q$  to fire only in an agent state", fontsize=FS_SMALL, color=MID)
    ax.text(150, yy + 69, r"up$_j = q$", fontsize=FS_SMALL, color=DARK)
    ax.text(150, yy + 85, r"down$_{:,j} = \pm\,\beta\rho\sqrt{n}\;\hat{u}\,/\,\mathrm{gelu}(G)$,   $\hat{u} = \mathrm{unit}(d\,/\,w_{\mathrm{post}})$", fontsize=FS_SMALL, color=DARK)
    ax.annotate("", xy=(140, yy + 42), xytext=(112, yy + 42), arrowprops=arrow)
    # output norm + effect
    box(420, yy + 22, 110, 40, "MLP output\nRMS norm, scale s", fc="#F7F7F7")
    ax.annotate("", xy=(420, yy + 42), xytext=(390, yy + 42), arrowprops=arrow)
    ax.annotate("", xy=(475, yy + 128), xytext=(475, yy + 62), arrowprops=dict(arrow, color=GREY))
    box(548, yy + 8, 146, 86, "", ec=GREEN, fc="#F1F8F3")
    ax.text(556, yy + 22, "predicted shift along d", fontsize=FS_SMALL + 0.3, fontweight="bold", color=GREEN)
    ax.text(556, yy + 44, r"$\Delta(\beta) = s\,\frac{\beta}{\sqrt{1+\beta^2}}\,\sqrt{n}\,\eta$", fontsize=FS_SMALL, color=DARK)
    ax.text(556, yy + 64, r"set $\beta$ so $\Delta = t$ class gaps", fontsize=FS_SMALL, color=DARK)
    ax.text(556, yy + 80, "realised: 0.96–1.14 × predicted", fontsize=FS_SMALL, color=DARK)
    ax.annotate("", xy=(548, yy + 50), xytext=(530, yy + 42), arrowprops=arrow)
    ax.text(W - 8, yy + 156, "9 tensors change · served as a standard LoRA by a stock engine · "
            "reproduces a neuron-addition hook (KL 0.0001 vs the edit's own 0.011)", fontsize=FS_SMALL - 0.3, color=MID, ha="right")
    finalize_figure(fig, "fig1_loop")


# ---------------------------------------------------------------------------------------------
# Figure 2: dose-response in decision replay, three edits
# ---------------------------------------------------------------------------------------------
DOSE = [
    dict(t="Submit sooner (on the RL adapter)", yl="Submit rate, 100 held-out decisions",
         xs=[0, .25, .5, 1], ys=[.15, .20, .29, .41], ci=[(.093, .233), (.133, .289), (.21, .385), (.319, .508)],
         ctl=[(.5, .15, (.093, .233))], ylim=(0, .55), fmt="{:.2f}", note="↑ more submits"),
    dict(t="Think briefly", yl="Mean thinking tokens per call, 320 decisions",
         xs=[0, .25, .5, 1], ys=[659, 594, 524, 399], ci=None,
         ctl=[(.5, 649, None)], ylim=(350, 700), fmt="{:.0f}", note="↓ shorter thinking"),
    dict(t="Explore more before editing", yl="First-edit rate, 320 held-out decisions",
         xs=[0, .125, .25, .5], ys=[.66, .61, .57, .49], ci=[(.61, .71), (.55, .66), (.51, .62), (.43, .54)],
         ctl=[(.5, .66, (.61, .71)), (.25, .66, None), (.25, .62, None), (.25, .64, None)], ylim=(.35, .75),
         fmt="{:.2f}", note="↓ fewer early edits"),
]


def fig2_dose():
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7))
    title(fig, "Each edit moves its target decision with dose; its matched random-direction control does not")
    for ax, p, tag in zip(axes, DOSE, "abc"):
        xs, ys = p["xs"], p["ys"]
        ax.plot(xs, ys, color=BLUE, lw=1.8, zorder=2, clip_on=False)
        if p["ci"]:
            for a, b, (lo, hi) in zip(xs, ys, p["ci"]):
                ax.errorbar([a], [b], yerr=[[b - lo], [hi - b]], color=BLUE, lw=0.9, capsize=2, zorder=1)
        ax.plot(xs, ys, zorder=3, clip_on=False, **dict(EDIT, ms=5.5))
        for a, b in zip(xs, ys):
            ax.annotate(p["fmt"].format(b), (a, b), xytext=(7, -3) if ys[-1] > ys[0] else (6, 4),
                        textcoords="offset points", fontsize=FS_SMALL, color=BLUE,
                        ha="left", va="top" if ys[-1] > ys[0] else "bottom")
        for k, (cx, cy, cci) in enumerate(p["ctl"]):
            dx = 0.035 * (k + 1)
            if cci:
                ax.errorbar([cx + dx], [cy], yerr=[[cy - cci[0]], [cci[1] - cy]], color=RED, lw=0.9, capsize=2, zorder=1)
            ax.plot([cx + dx], [cy], zorder=4, clip_on=False, **dict(CTRL, ms=4.8))
        cx, cy, _ = p["ctl"][0]
        if len(p["ctl"]) == 1:
            ax.annotate("random-direction\ncontrol", (cx + 0.035, cy), xytext=(7, 0), textcoords="offset points",
                        fontsize=FS_SMALL, color=RED, va="center", fontweight="bold", linespacing=1.0)
        else:
            ax.text(0.03, 0.05, "4 random-direction controls", transform=ax.transAxes, fontsize=FS_SMALL,
                    color=RED, fontweight="bold", va="bottom", ha="left")
        ax.set_xlim(-0.06, xs[-1] * 1.30)
        ax.set_ylim(*p["ylim"])
        ax.set_xticks(xs); ax.set_xticklabels(["0\nunedited"] + [f"{v:g}" for v in xs[1:]])
        ax.set_xlabel("Edit strength (class gaps)")
        ax.set_ylabel(p["yl"], fontsize=FS_SMALL)
        ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=FS + 1.5, fontweight="bold", ha="right", va="bottom")
        ax.text(0.03, 1.04, p["t"], transform=ax.transAxes, fontsize=FS, fontweight="bold", ha="left", va="bottom")
        ax.text(0.98, 0.04 if len(p['ctl']) == 1 else 0.13, p["note"], transform=ax.transAxes, fontsize=FS_SMALL, color=MID, ha="right", va="bottom")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.21, wspace=0.42)
    finalize_figure(fig, "fig2_dose")


# ---------------------------------------------------------------------------------------------
# Figures 3, 4 and 6: one row per arm
# ---------------------------------------------------------------------------------------------
def dot_rows(name, head, rows, xlim, xticks, xfmt, xlabel, refs, height, ci=False, stack=0.22, extra=None):
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    left, right = 0.285, 0.905
    fig.subplots_adjust(left=left, right=right, top=1 - 0.52 / height, bottom=0.50 / height)
    title(fig, head)
    t_lab = blended_transform_factory(fig.transFigure, ax.transData)
    ax.set_xlim(*xlim); ax.set_ylim(len(rows) - 0.5, -0.5)
    for v in xticks:
        ax.axvline(v, color=LIGHT, lw=0.5, zorder=0)
    for v, lab, col, ls in refs:
        ax.axvline(v, color=col, lw=1.1, ls=ls, zorder=1)
        ax.annotate(lab, (v, -0.5), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_SMALL, color=col, annotation_clip=False, fontweight="bold")
    for i, (label, n, vals, kind) in enumerate(rows):
        style, col = KIND[kind]
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#F6F6F6", lw=0, zorder=-1)
        m = sum(vals) / len(vals)
        if ci and len(vals) >= 2:
            sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            h = T95[len(vals)] * sd / len(vals) ** 0.5
            ax.plot([m - h, m + h], [i, i], color=col, lw=4.5, alpha=0.22, solid_capstyle="butt", zorder=1)
        ax.plot([m, m], [i - 0.36, i + 0.36], color=col, lw=2.4, solid_capstyle="butt", zorder=2)
        seen = {}
        for v in vals:
            k = seen.get(v, 0); seen[v] = k + 1
            vv = min(max(v, xlim[0] + 0.3), xlim[1] - 0.3)
            ax.plot([vv], [i + k * stack - stack / 2 * (vals.count(v) - 1)], zorder=3, **style)
        clipped = [v for v in vals if v < xlim[0] + 0.3 or v > xlim[1] - 0.3]
        if clipped:
            ax.annotate(", ".join(f"{v:+d}".replace("-", "−") for v in clipped) + "  (off scale: breaks the model)",
                        (xlim[0] + 0.3, i), xytext=(8, 0), textcoords="offset points", fontsize=FS_SMALL,
                        color=col, va="center")
        ax.text(0.012, i - 0.14, label, transform=t_lab, ha="left", va="center", fontsize=FS,
                fontweight="bold" if kind == "e" else "normal", color=DARK)
        ax.text(0.012, i + 0.24, n, transform=t_lab, ha="left", va="center", fontsize=FS_SMALL - 0.5, color=MID)
        ax.text(0.988, i, (f"{m:+.1f}" if xlim[0] < 0 else f"{m:.1f}").replace("-", "−"), transform=t_lab,
                ha="right", va="center", fontsize=FS, fontweight="bold", color=col)
    ax.annotate("mean", (0.988, -0.5), xycoords=t_lab, xytext=(0, 3), textcoords="offset points", ha="right",
                va="bottom", fontsize=FS_SMALL, color=MID, annotation_clip=False)
    ax.set_xticks(xticks); ax.set_xticklabels([xfmt(v) for v in xticks])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=4)
    if extra:
        extra(fig, ax)
    finalize_figure(fig, name)


SOLVE_ROWS = [
    ("Unedited model", "13 runs", [28, 24, 22, 29, 22, 29, 27, 26, 25, 27, 30, 28, 25], "n"),
    ("Matched random controls", "14 runs, 4 adapters", [27, 23, 26, 27, 27, 25, 28, 24, 23, 27, 25, 25, 25, 26], "c"),
    ("Explore-more, strength 0.125", "4 runs", [31, 30, 28, 33], "e"),
    ("Explore-more, strength 0.25", "7 runs", [32, 29, 30, 31, 29, 26, 27], "e"),
    ("Explore-more, strength 0.5", "10 runs", [28, 31, 33, 30, 26, 32, 30, 24, 28, 27], "e"),
    ("Explore-more, strength 1.0", "1 run", [18], "e"),
]

ARENA_ROWS = [
    ("Before-first-edit gate", "Claude · 6 batches", [9, 2, 3, 9, 5, 2], "e"),
    ("Early-calls gate", "Claude · 6 batches", [6, 2, 0, 9, 7, 4], "e"),
    ("Claude’s random control", "4 batches", [1, 1, 4, 1], "c"),
    ("Decision-token gate + rotation", "GPT · 6 batches", [4, 0, 2, 8, 2, -2], "e"),
    ("GPT’s random control", "4 batches", [-1, -3, 6, 0], "c"),
    ("Explore + concise", "Gemini · 3 batches", [7, 3, 7], "e"),
    ("Gemini’s random control", "3 batches", [8, -3, 4], "c"),
]

# Round 4: first and second batch of each arm against the unedited arm of its own batch
# (first: b45, b36 [stopped at 156 tasks], b43, b44; second: b47-b50). Source: parent session finals 4:06 / 4:14 PM.
R4 = [("pe_L5", "explore gate, 5 layers", [5, 5]), ("pe_lo", "explore gate, low dose", [4, 6]),
      ("pe_L20", "explore gate, layer 20", [1, 8]), ("u36", "explore + commit late", [1, 9]),
      ("u63", "explore + commit late", [3, 4]), ("late60", "commit late", [6, 0]),
      ("pre_early", "gate + early calls", [3, 2]), ("late30", "commit late", [2, 2]),
      ("pe_c6", "explore gate, from call 6", [-1, 5]), ("u33_L5", "explore + commit late, 5 layers", [-2, 5]),
      ("pre15", "before-first-edit gate", [5, -2]), ("pe_hi", "explore gate, high dose", [-2, 3]),
      ("u33_c6", "explore + commit late", [-4, 5]), ("pre35", "before-first-edit gate", [3, -3]),
      ("pu", "explore + commit late", [3, -3]), ("u33", "explore + commit late", [-4, 2]),
      ("pe_L22", "explore gate, layer 22", [1, -4]), ("pe_L14", "explore gate, layer 14", [-22, -28])]
R4_CTRL = [("u33_ctrl", "random control", [7, 1]), ("pe_ctrl", "random control", [-4, 0])]


def fig3_solve():
    dot_rows("fig3_solve", "Low doses of always-on explore-more are ahead of the unedited model and its controls; 1.0 overshoots",
             SOLVE_ROWS, (16, 36), list(range(16, 37, 4)), str,
             "Tasks solved of 170 (one marker per full evaluation run; bar = mean)",
             [(26.3, "unedited mean 26.3", MID, (0, (3, 2.5)))], height=3.2)


def fig4_arena():
    dot_rows("fig4_arena", "Three LLM authors: only Claude's two gates have intervals clear of zero; controls also drift upward",
             ARENA_ROWS, (-4, 10), list(range(-4, 11, 2)), lambda v: f"{v:+d}".replace("-", "−") if v else "0",
             "Solved tasks minus the unedited arm of the same batch (marker = batch; bar = mean; shade = 95% t CI)",
             [(0, "unedited", MID, (0, (3, 2.5)))], height=3.45, ci=True)


def fig6_round4():
    r4 = sorted(R4[:-1], key=lambda d: -sum(d[2])) + R4[-1:]
    rows = [(n, d, v, "e") for n, d, v in r4] + [(n, d, v, "c") for n, d, v in R4_CTRL]
    rows = [(f"{n}", d, v, k) for n, d, v, k in rows]
    def extra(fig, ax):
        ax.axvspan(8, 12, color=GREEN, alpha=0.07, zorder=0)
        ax.text(10, 0.6, "pass zone\n(mean ≥ +8)", ha="center", va="center", fontsize=FS_SMALL, color=GREEN)
    dot_rows("fig6_round4", "Round 4, pre-registered screen of 18 edits and 2 random controls: the best edits average +5 tasks",
             rows, (-12, 12), list(range(-12, 13, 4)), lambda v: f"{v:+d}".replace("-", "−") if v else "0",
             "Solved tasks minus the unedited arm of the same batch, 170 tasks (marker = batch; bar = mean of two)",
             [(0, "unedited", MID, (0, (3, 2.5)))], height=5.6, stack=0.22)


def fig5_explore():
    bins = ["0–2", "3–4", "5–7", "8+"]
    orr = [0.76, 1.42, 1.72, 0.12]; lo = [0.39, 0.86, 0.97, 0.04]; hi = [1.48, 2.35, 3.06, 0.38]
    share = [27, 34, 22, 17]
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    fig.text(0.02, 0.985, "Episodes that explore 8+ calls before\nthe first edit are the ones lost",
             ha="left", va="top", fontsize=FS + 0.5, fontweight="bold")
    for i in range(4):
        col = RED if hi[i] < 1 else GREY
        ax.errorbar([i], [orr[i]], yerr=[[orr[i] - lo[i]], [hi[i] - orr[i]]], fmt="o", color=col, mfc=col,
                    mec="white", ms=6.5, lw=1.3, capsize=3)
        ax.annotate(f"{orr[i]:.2f}", (i, orr[i]), xytext=(9, 0), textcoords="offset points", fontsize=FS_SMALL,
                    color=col, va="center", fontweight="bold" if col == RED else "normal")
    ax.axhline(1, color=MID, lw=0.9, ls=(0, (3, 2.5)))
    ax.text(3.45, 1.08, "no association", fontsize=FS_SMALL - 0.5, color=MID, ha="right", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(0.03, 4.5)
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{b}\n{s}% of ep." for b, s in zip(bins, share)], fontsize=FS_SMALL); ax.set_xlim(-0.5, 3.6)
    ax.set_xlabel("Tool calls before the first file edit", labelpad=2)
    ax.set_ylabel("Odds of solving vs the task's\nother episodes (95% CI, log)", fontsize=FS_SMALL)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.24)
    finalize_figure(fig, "fig5_invu")


if __name__ == "__main__":
    apply_publication_style()
    fig1_loop(); fig2_dose(); fig3_solve(); fig4_arena(); fig5_explore(); fig6_round4()
    print("ok")
