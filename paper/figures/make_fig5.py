"""Figure 5: within-task Mantel-Haenszel odds ratio of solving by calls before the first file edit (13 unedited runs, b3-b15,
2,210 episodes), each bin against all other episodes of the same task, with 95% Robins-Breslow-Greenland intervals.
Source: diag_recompute/result.json (independent recomputation; matches the authoring model's diag2 output)."""
import matplotlib.pyplot as plt
from make_figures import OUT, BLUE, GREY, DARK, FS_SMALL, apply_publication_style
apply_publication_style()
bins = ["0-2", "3-4", "5-7", "8+"]
oratio = [0.76, 1.42, 1.72, 0.12]
lo = [0.39, 0.86, 0.97, 0.04]
hi = [1.48, 2.35, 3.06, 0.38]
share = [0.271, 0.341, 0.219, 0.169]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = range(4)
ax.errorbar(x, oratio, yerr=[[o - l for o, l in zip(oratio, lo)], [h - o for o, h in zip(oratio, hi)]],
            fmt="o", color=BLUE, mfc=BLUE, mec="white", ms=6, lw=1.2, capsize=3)
ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
for i, (o, s) in enumerate(zip(oratio, share)):
    ax.annotate(f"{o:.2f}", (i, o), textcoords="offset points", xytext=(12, -3), ha="left", fontsize=FS_SMALL, color=DARK)
    ax.annotate(f"{s:.0%}", (i, 0), textcoords="offset points", xytext=(0, -4), ha="center", va="top", fontsize=FS_SMALL - 1, color=GREY)
ax.set_yscale("log")
ax.set_yticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4], ["0.05", "0.1", "0.25", "0.5", "1", "2", "4"])
ax.set_ylim(0.03, 4.5)
ax.set_xticks(list(x), bins)
ax.set_xlabel("Calls before the first file edit (grey: share of episodes)")
ax.set_ylabel("Within-task odds ratio\nof solving (95% CI)")
fig.tight_layout()
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT / f"fig5_invu.{ext}", dpi=300)
