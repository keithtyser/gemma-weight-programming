"""Web figures for the Kaggle Writeup. The Writeup shows images about 500 px wide, so every figure is drawn at
6.4 in wide with 15-17 pt text (about 16 px on screen), one message per figure, direct labels, no legends.
Outputs web_*.png (200 dpi) next to this file. Data are the same as make_figures_v2.py (sources noted there)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent
BLUE, RED, GREY, DARK, MID, LIGHT, GREEN = "#1F5FA8", "#C0392B", "#8A8A8A", "#222222", "#555555", "#DDDDDD", "#2E7D4F"
AMBER = "#B8660B"
W = 6.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 15, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.3,
                     "axes.edgecolor": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.labelsize": 15,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / f"web_{name}.png", dpi=200)
    plt.close(fig)


def head(fig, text, sub=None):
    fig.text(0.03, 0.975, text, ha="left", va="top", fontsize=18, fontweight="bold", color=DARK)
    if sub:
        fig.text(0.03, 0.905, sub, ha="left", va="top", fontsize=13.5, color=MID)


# 1. the loop: a 2 x 3 snake (row 1 left to right, row 2 right to left), dashed return arrow
def loop():
    fig = plt.figure(figsize=(W, 5.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(114, 0); ax.axis("off")
    head(fig, "The loop: an LLM edits an agent's weights", "No gradients. Only step 5 (orange) uses a GPU.")
    # who acts on what: author chip (left) writes a LoRA adapter for the frozen agent (right)
    ax.add_patch(FancyBboxPatch((3, 21), 40, 12, boxstyle="round,pad=0,rounding_size=6", fc=DARK, ec="none"))
    ax.text(23, 27, "LLM author\nClaude · GPT · Gemini", ha="center", va="center", fontsize=12.5, color="white",
            fontweight="bold", linespacing=1.1)
    ax.add_patch(FancyBboxPatch((57, 21), 40, 12, boxstyle="round,pad=0,rounding_size=6", fc="white", ec=DARK, lw=1.8))
    ax.text(77, 27, "Gemma 4 E4B agent\n+ LoRA adapter (the edit)", ha="center", va="center", fontsize=12.5,
            color=DARK, fontweight="bold", linespacing=1.1)
    ax.annotate("", xy=(56.5, 27), xytext=(43.5, 27),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.2, mutation_scale=16))
    ax.text(50, 24.2, "writes", ha="center", va="bottom", fontsize=12, color=MID, style="italic")
    ax.plot([3, 97], [37.5, 37.5], color=LIGHT, lw=1.2)
    steps = {1: ("1 Diagnose", "where are\nepisodes lost?"), 2: ("2 Locate", "direction for\nthat decision"),
             3: ("3 Write", "neuron edit\n+ random control"), 4: ("4 Predict", "expected effect,\nwritten first"),
             5: ("5 Verify", "replay + run vs\nunedited model"), 6: ("6 Score", "was the\nprediction right?")}
    pos = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (2, 1), 5: (1, 1), 6: (0, 1)}
    bw, bh, x0, y0, dx, dy = 28, 27, 3, 43, 33.5, 40
    centre = {}
    for k, (h, b) in steps.items():
        c, r = pos[k]; x, y = x0 + c * dx, y0 + r * dy
        centre[k] = (x + bw / 2, y + bh / 2)
        gpu = k == 5; ec = AMBER if gpu else BLUE
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc="#FDF3E6" if gpu else "#EEF3FA", ec=ec, lw=2))
        ax.text(x + 2.4, y + 7.5, h, fontsize=16, fontweight="bold", color=ec, va="center")
        ax.text(x + 2.4, y + 18.5, b, fontsize=13.5, color=DARK, va="center", linespacing=1.15)
    A = dict(arrowstyle="-|>", color=MID, lw=2.2, mutation_scale=16)
    def arr(xy, xyt, **kw):
        ax.annotate("", xy=xy, xytext=xyt, arrowprops=dict(A, **kw))
    arr((x0 + dx - 0.6, y0 + bh / 2), (x0 + bw + 0.6, y0 + bh / 2))
    arr((x0 + 2 * dx - 0.6, y0 + bh / 2), (x0 + dx + bw + 0.6, y0 + bh / 2))
    arr((x0 + 2 * dx + bw / 2, y0 + dy - 0.6), (x0 + 2 * dx + bw / 2, y0 + bh + 0.6))
    arr((x0 + dx + bw + 0.6, y0 + dy + bh / 2), (x0 + 2 * dx - 0.6, y0 + dy + bh / 2))
    arr((x0 + bw + 0.6, y0 + dy + bh / 2), (x0 + dx - 0.6, y0 + dy + bh / 2))
    arr((x0 + bw / 2, y0 + bh + 0.6), (x0 + bw / 2, y0 + dy - 0.6), ls="--", color=GREY)
    ax.text(x0 + bw / 2 + 2, y0 + bh + 6.5, "iterate", fontsize=13, color=MID, style="italic", va="center")
    save(fig, "fig1_loop")


# 2. exploration odds ratios
def explore():
    fig, ax = plt.subplots(figsize=(W, 4.8))
    head(fig, "Long exploration is where episodes are lost", "Odds of solving vs the same task's other episodes (95% CI)")
    bins = ["0–2", "3–4", "5–7", "8+"]; orr = [0.76, 1.42, 1.72, 0.12]
    lo = [0.39, 0.86, 0.97, 0.04]; hi = [1.48, 2.35, 3.06, 0.38]
    for i in range(4):
        col = RED if hi[i] < 1 else GREY
        ax.errorbar([i], [orr[i]], yerr=[[orr[i] - lo[i]], [hi[i] - orr[i]]], fmt="o", color=col, ms=11,
                    mec="white", mew=1.5, lw=2.2, capsize=6)
        ax.text(i + 0.14, orr[i], f"{orr[i]:.2f}", va="center", fontsize=15, color=col,
                fontweight="bold" if col == RED else "normal")
    ax.annotate("8+ calls: odds of\nsolving ~8× lower", xy=(2.88, 0.12), xytext=(1.6, 0.085), fontsize=13.5,
                color=RED, ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.axhline(1, color=MID, lw=1.3, ls="--")
    ax.text(3.45, 1.12, "no effect", ha="right", fontsize=13, color=MID)
    ax.text(-0.33, 3.3, "↑ more likely solved", fontsize=12.5, color=MID, va="center")
    ax.text(-0.33, 0.042, "↓ less likely solved", fontsize=12.5, color=MID, va="center")
    ax.text(1.0, 0.36, "grey: CI includes 1", fontsize=12.5, color=GREY, ha="center")
    ax.set_yscale("log"); ax.set_ylim(0.03, 4.5); ax.minorticks_off()
    ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
    ax.set_xticks(range(4)); ax.set_xlim(-0.4, 3.55)
    ax.set_xticklabels([f"{b}\n{s}% of ep." for b, s in zip(bins, [27, 34, 22, 17])], fontsize=13)
    ax.set_xlabel("Tool calls before the first file edit")
    ax.set_ylabel("Odds ratio (log)")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.22)
    save(fig, "fig2_explore")


# 3. dose: one panel, all three edits as % movement toward the targeted behaviour
def dose():
    fig, ax = plt.subplots(figsize=(W, 4.6))
    head(fig, "Edits move their decision with dose; controls don't",
         "Change in the targeted decision vs the unedited model, held-out replay")
    series = [("submit sooner", BLUE, [0, 33, 93, 173], [(2, 0)]),
              ("think briefly", "#2A9D8F", [0, 10, 20, 39], [(2, 2)]),
              ("explore first", "#7B4FA0", [0, 8, 14, 26], [(2, 0), (2, 6), (2, 3), (3, 0)])]
    xs = [0, 1, 2, 3]
    for name, col, ys, ctl in series:
        ax.plot(xs, ys, "-o", color=col, lw=2.6, ms=9, mec="white", mew=1.5)
        ax.text(3.12, ys[-1] + {39: 5, 26: -5}.get(ys[-1], 0), f"{name}  +{ys[-1]}%", va="center", fontsize=14, color=col, fontweight="bold")
        for cx, c in ctl:
            ax.plot([cx], [c], "s", mfc="white", mec=RED, mew=2, ms=9, zorder=5)
    ax.text(2.5, -15, "random controls (□): 0 to 6%", fontsize=13.5, color=RED, ha="center", va="center")
    ax.axhline(0, color=MID, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(["unedited", "low", "mid", "high"])
    ax.set_xlabel("Edit strength")
    ax.set_ylabel("Toward targeted behaviour (%)")
    ax.set_xlim(-0.2, 5.0); ax.set_ylim(-26, 185)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.80, bottom=0.15)
    save(fig, "fig3_dose")


def rows_plot(name, title, sub, rows, xlim, xticks, xlabel, ref, refl, h, rule=None, link=False, shade=None):
    fig, ax = plt.subplots(figsize=(W, h))
    head(fig, title, sub)
    n = len(rows)
    for i, (lab, vals, kind) in enumerate(rows):
        col = {"e": BLUE, "c": RED, "n": GREY}[kind]
        if (i % 2 == 0) if shade is None else (i in shade):
            ax.axhspan(i - 0.5, i + 0.5, color="#F4F4F4", lw=0, zorder=0)
        m = sum(vals) / len(vals)
        if link and len(vals) > 1:
            ax.plot([max(min(vals), xlim[0] + 0.2), max(vals)], [i, i], color=col, lw=1.2, alpha=0.35, zorder=2)
        seen = {}
        for v in vals:
            vv = max(v, xlim[0] + 0.2)
            k = seen.get(v, 0); seen[v] = k + 1
            yy = i + (k - (vals.count(v) - 1) / 2) * 0.24
            ax.plot([vv], [yy], "o" if kind != "c" else "s", ms=8 if kind != "c" else 7.5, mfc=col if kind != "c" else "white",
                    mec=col if kind == "c" else "white", mew=1.6, alpha=0.75, zorder=3)
        ax.plot([m, m], [i - 0.34, i + 0.34], color=col, lw=4, zorder=4, solid_capstyle="butt")
        ax.text(xlim[1] + (xlim[1] - xlim[0]) * 0.02, i, (f"{m:+.1f}" if xlim[0] < 0 else f"{m:.1f}").replace("-", "−"),
                va="center", fontsize=14, color=col, fontweight="bold")
    ax.axvline(ref, color=MID, lw=1.3, ls="--", zorder=1)
    ax.text(ref, -0.75, refl, ha="center", va="bottom", fontsize=13, color=MID)
    if rule is not None:
        ax.axvspan(rule, xlim[1], color=GREEN, alpha=0.10, zorder=0)
        ax.axvline(rule, color=GREEN, lw=2, zorder=1)
        ax.text(rule - 0.3, -0.75, "pass bar: mean ≥ +8", ha="right", va="bottom", fontsize=13, color=GREEN, fontweight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color({"e": DARK, "c": RED, "n": MID}[r[2]]); t.set_fontweight("bold" if r[2] == "e" else "normal")
    ax.set_ylim(n - 0.5, -0.9)
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_xticklabels([(f"{v:+d}" if xlim[0] < 0 and v else str(v)).replace("-", "−") for v in xticks])
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    return fig, ax


def solve():
    rows = [("unedited (13 runs)", [28, 24, 22, 29, 22, 29, 27, 26, 25, 27, 30, 28, 25], "n"),
            ("random controls (14)", [27, 23, 26, 27, 27, 25, 28, 24, 23, 27, 25, 25, 25, 26], "c"),
            ("explore-more 0.125 (4)", [31, 30, 28, 33], "e"), ("explore-more 0.25 (7)", [32, 29, 30, 31, 29, 26, 27], "e"),
            ("explore-more 0.5 (10)", [28, 31, 33, 30, 26, 32, 30, 24, 28, 27], "e"), ("explore-more 1.0 (1)", [18], "e")]
    fig, ax = rows_plot("solve", "Low doses are ahead of unedited and controls", "Tasks solved of 170 per run (dot = run, bar = mean, (n) = runs)",
                        rows, (16, 35), [16, 20, 24, 28, 32], "Tasks solved of 170", 26.3, "unedited mean", 4.6)
    fig.subplots_adjust(left=0.36, right=0.88, top=0.78, bottom=0.15)
    save(fig, "fig4_solve")


def arena():
    rows = [("Claude: first-edit gate", [9, 2, 3, 9, 5, 2], "e"), ("Claude: early-calls gate", [6, 2, 0, 9, 7, 4], "e"),
            ("Claude's random control", [1, 1, 4, 1], "c"),
            ("GPT: gate + rotation", [4, 0, 2, 8, 2, -2], "e"), ("GPT's random control", [-1, -3, 6, 0], "c"),
            ("Gemini: explore + concise", [7, 3, 7], "e"), ("Gemini's random control", [8, -3, 4], "c")]
    fig, ax = rows_plot("arena", "Three LLM authors, one brief", "Solved tasks minus unedited, same batch (dot = batch, bar = mean)",
                        rows, (-4, 10), [-4, 0, 4, 8], "Tasks gained vs unedited (of 170)", 0, "unedited", 4.9,
                        shade={0, 1, 2, 5, 6})
    fig.subplots_adjust(left=0.40, right=0.88, top=0.79, bottom=0.14)
    save(fig, "fig5_arena")


def round4():
    data = [("pe_L5", [5, 5]), ("pe_lo", [4, 6]), ("pe_L20", [1, 8]), ("u36", [1, 9]), ("u63", [3, 4]),
            ("late60", [6, 0]), ("pre_early", [3, 2]), ("late30", [2, 2]), ("pe_c6", [-1, 5]), ("u33_L5", [-2, 5]),
            ("pre15", [5, -2]), ("pe_hi", [-2, 3]), ("u33_c6", [-4, 5]), ("pre35", [3, -3]), ("pu", [3, -3]),
            ("u33", [-4, 2]), ("pe_L22", [1, -4]), ("pe_L14 (broke)", [-22, -28])]
    data = sorted(data[:-1], key=lambda d: -sum(d[1])) + data[-1:]
    rows = [(n, v, "e") for n, v in data] + [("control u33_ctrl", [7, 1], "c"), ("control pe_ctrl", [-4, 0], "c")]
    fig, ax = rows_plot("round4", "Round 4: no edit reaches the +8 bar", "20 adapters, two batches each (bar = mean)",
                        rows, (-8, 12), [-8, -4, 0, 4, 8, 12], "Tasks gained vs unedited (of 170)", 0, "unedited", 7.6, rule=8, link=True)
    ax.text(-7.6, 17, "−22, −28 off scale", va="center", ha="left", fontsize=12, color=BLUE)
    fig.subplots_adjust(left=0.30, right=0.87, top=0.855, bottom=0.08)
    save(fig, "fig6_round4")


if __name__ == "__main__":
    loop(); explore(); dose(); solve(); arena(); round4()
    print("ok")
