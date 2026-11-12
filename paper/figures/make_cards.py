"""Editorial-card versions of the six web figures for the Kaggle Writeup.
Data, labels and numbers are taken from make_figures_web.py (same values). Each figure is an HTML page with an
inline SVG, rendered by headless Chrome at 1200 CSS px x deviceScaleFactor 2 (2400 px wide), then cropped to the
card plus a page-coloured margin with PIL. Light theme -> card_light_fig*.png; dark theme (Kaggle page
background #181818) -> card_fig*.png."""
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageChops

OUT = Path(__file__).resolve().parent
CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe"

THEMES = {
    "light": dict(PAGE="#FFFFFF", HEAD="#2B2118", INK="#2B2118", MUTED="#7A6A5A", CARD="#FBF4EA", BORDER="#E8D9C4",
                  BAND="#F3E6D3", TILE="#FFFAF3", INDIGO="#3D4FC4", ORANGE="#D9622B", GREY="#9A8F84", TEAL="#2A8C82",
                  PLUM="#8A4F9E", GREEN="#4E8A5B", STEPFILL="#EEF0FA", GPUFILL="#FBE6D6", HOTBAND="#F7DCCB",
                  ZONE="#E1EDDD", GRID="#E8D9C4", CHIP="#2B2118", CHIPTXT="#FFFFFF", CHIPSUB="#E9DCCB"),
    "dark": dict(PAGE="#181818", HEAD="#FFFFFF", INK="#F2EEE8", MUTED="#CFC9BF", CARD="#222222", BORDER="#3A3A3A",
                 BAND="#2B2B2B", TILE="#1B1B1B", INDIGO="#A8B4FF", ORANGE="#FFA266", GREY="#CBC6BD", TEAL="#62E3CF",
                 PLUM="#DDB3F7", GREEN="#8FD9A0", STEPFILL="#262B4A", GPUFILL="#3D2A1D", HOTBAND="#36281F",
                 ZONE="#26352A", GRID="#3A3A3A", CHIP="#F2EEE8", CHIPTXT="#141414", CHIPSUB="#3A352E"),
}


def set_theme(name):
    globals().update(THEMES[name])


set_theme("light")
SERIF = "Georgia, 'Times New Roman', serif"
SANS = "'Segoe UI', Arial, Helvetica, sans-serif"
MONO = "Consolas, 'Courier New', monospace"
SW = 1072  # SVG width inside the card (1200 - 2*24 margin - 2*40 padding)

CSS_T = """
* { box-sizing: border-box; margin: 0; padding: 0; font-variant-numeric: lining-nums; font-feature-settings: "lnum" 1; }
html, body { background: %(PAGE)s; }
body { width: 1200px; padding: 24px; font-family: %(SANS)s; color: %(INK)s; }
.card { background: %(CARD)s; border: 1px solid %(BORDER)s; border-radius: 10px; padding: 34px 40px 26px; }
h1 { font-family: %(SERIF)s; font-weight: bold; font-size: 34px; line-height: 1.2; color: %(HEAD)s; letter-spacing: -0.2px; }
.sub { font-size: 21px; color: %(MUTED)s; margin-top: 8px; line-height: 1.35; }
.tiles { display: flex; gap: 16px; margin-top: 24px; }
.tile { flex: 1; background: %(TILE)s; border: 1px solid %(BORDER)s; border-radius: 8px; padding: 14px 18px 14px;
        position: relative; overflow: hidden; }
.tile.bad::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: %(ORANGE)s; }
.tile .lab { font-family: %(MONO)s; font-size: 18px; letter-spacing: 0.6px; text-transform: uppercase; color: %(MUTED)s; }
.tile .num { font-family: %(SERIF)s; font-size: 46px; line-height: 1.1; margin-top: 6px; color: %(HEAD)s; }
.tile.bad .num { color: %(ORANGE)s; }
.tile .num small { font-size: 24px; color: %(MUTED)s; }
.tile .sl { font-family: %(MONO)s; font-size: 19px; color: %(MUTED)s; margin-top: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 26px; margin-top: 20px; font-family: %(MONO)s; font-size: 19px;
          color: %(INK)s; align-items: center; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.chart { margin-top: 14px; }
.foot { font-family: %(MONO)s; font-size: 19px; color: %(MUTED)s; margin-top: 14px; padding-top: 12px;
        border-top: 1px solid %(BORDER)s; line-height: 1.4; }
"""


# ---------- small helpers ----------
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def minus(s):
    return s.replace("-", "\u2212")


def T(x, y, s, size=16, fill=None, anchor="start", weight="normal", family=None, base="central", extra=""):
    fill = fill or INK
    size = size + 1.5 if size < 20 else size
    fam = f' font-family="{family or SANS}"' 
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" dominant-baseline="{base}"{fam} {extra}>{esc(s)}</text>')


def tiles(items):
    out = []
    for lab, num, sl, bad in items:
        out.append(f'<div class="tile{" bad" if bad else ""}"><div class="lab">{esc(lab)}</div>'
                   f'<div class="num">{num}</div><div class="sl">{esc(sl)}</div></div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def glyph(kind, col):
    if kind == "dot":
        return f'<svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="{col}"/></svg>'
    if kind == "ring":
        return f'<svg width="14" height="14"><circle cx="7" cy="7" r="5.2" fill="none" stroke="{col}" stroke-width="2"/></svg>'
    if kind == "sq":
        return f'<svg width="14" height="14"><rect x="2" y="2" width="10" height="10" fill="{CARD}" stroke="{col}" stroke-width="2"/></svg>'
    if kind == "bar":
        return f'<svg width="10" height="18"><rect x="3" y="0" width="4" height="18" fill="{col}"/></svg>'
    if kind == "dash":
        return f'<svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke="{col}" stroke-width="1.6" stroke-dasharray="5 4"/></svg>'
    if kind == "line":
        return f'<svg width="26" height="12"><line x1="0" y1="6" x2="26" y2="6" stroke="{col}" stroke-width="3"/><circle cx="13" cy="6" r="4.5" fill="{col}"/></svg>'
    if kind == "zone":
        return f'<svg width="22" height="16"><rect x="0" y="0" width="22" height="16" rx="2" fill="{ZONE}" stroke="{GREEN}" stroke-width="1.2"/></svg>'
    raise ValueError(kind)


def legend(items):
    return '<div class="legend">' + "".join(f"<span>{glyph(k, c)}{esc(t)}</span>" for k, c, t in items) + "</div>"


def page(title, sub, body):
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS_T % globals()}</style></head><body><div class="card">'
            f'<h1>{esc(title)}</h1><div class="sub">{esc(sub)}</div>{body}</div></body></html>')


def render(html, name):
    prefix = "card_light_" if PAGE == "#FFFFFF" else "card_"
    tmp = Path(tempfile.mkdtemp())
    hp = tmp / f"{name}.html"
    hp.write_text(html, encoding="utf-8")
    raw = tmp / f"{name}_raw.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=1200,2600", f"--screenshot={raw}", hp.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    im = Image.open(raw).convert("RGB")
    bg = Image.new("RGB", im.size, tuple(int(PAGE[i:i + 2], 16) for i in (1, 3, 5)))
    box = ImageChops.difference(im, bg).getbbox()
    assert box is not None and box[3] < im.size[1] - 4, f"{name}: card clipped, increase window height"
    pad = 48  # 24 CSS px of white around the card
    im = im.crop((0, max(0, box[1] - pad), im.size[0], min(im.size[1], box[3] + pad)))
    im.save(OUT / f"{prefix}{name}.png")
    print(prefix + name, im.size)


# ---------- 1. loop ----------
def loop():
    H = 482
    s = [f'<svg width="{SW}" height="{H}" viewBox="0 0 {SW} {H}">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
         f'<marker id="ahd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>']
    # lane: author writes adapter
    cw, ch, ly = 420, 70, 10
    s.append(f'<rect x="0" y="{ly}" width="{cw}" height="{ch}" rx="35" fill="{CHIP}"/>')
    s.append(T(cw / 2, ly + 24, "LLM author", 17, CHIPSUB, "middle", family=MONO, extra='letter-spacing="1"'))
    s.append(T(cw / 2, ly + 48, "Claude · GPT · Gemini", 22, CHIPTXT, "middle", "bold", family=SERIF))
    ax = SW - cw
    s.append(f'<rect x="{ax}" y="{ly}" width="{cw}" height="{ch}" rx="35" fill="{TILE}" stroke="{INK}" stroke-width="1.6"/>')
    s.append(T(ax + cw / 2, ly + 24, "Gemma 4 E4B agent", 20, HEAD, "middle", "bold", family=SERIF))
    s.append(T(ax + cw / 2, ly + 48, "+ LoRA adapter (the edit)", 17, MUTED, "middle", family=MONO))
    s.append(f'<line x1="{cw + 14}" y1="{ly + ch / 2}" x2="{ax - 14}" y2="{ly + ch / 2}" stroke="{INK}" '
             f'stroke-width="2.2" marker-end="url(#ahd)"/>')
    s.append(T(SW / 2, ly + ch / 2 - 16, "writes", 17, MUTED, "middle", family=MONO))
    s.append(f'<line x1="0" y1="{ly + ch + 26}" x2="{SW}" y2="{ly + ch + 26}" stroke="{BORDER}" stroke-width="1.6"/>')
    # steps: 2 x 3 snake
    steps = {1: ("Diagnose", "where are", "episodes lost?"), 2: ("Locate", "direction for", "that decision"),
             3: ("Write", "neuron edit", "+ random control"), 4: ("Predict", "expected effect,", "written first"),
             5: ("Verify", "replay + run vs", "unedited model"), 6: ("Score", "was the", "prediction right?")}
    pos = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (2, 1), 5: (1, 1), 6: (0, 1)}
    bw, bh, gx, gy, y0 = 300, 128, (SW - 900) / 2, 72, 136
    X = lambda c: c * (bw + gx)
    Y = lambda r: y0 + r * (bh + gy)
    for k, (h, b1, b2) in steps.items():
        c, r = pos[k]; x, y = X(c), Y(r)
        gpu = k == 5
        ec, fc = (ORANGE, GPUFILL) if gpu else (INDIGO, STEPFILL)
        s.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="9" fill="{fc}" stroke="{ec}" stroke-width="1.8"/>')
        s.append(f'<circle cx="{x + 30}" cy="{y + 34}" r="16" fill="{ec}"/>')
        s.append(T(x + 30, y + 35, str(k), 18, CARD, "middle", "bold", family=SERIF))
        s.append(T(x + 56, y + 35, h, 24, ec, "start", "bold", family=SERIF))
        if gpu:
            s.append(f'<rect x="{x + bw - 66}" y="{y + 22}" width="50" height="26" rx="4" fill="{ORANGE}"/>')
            s.append(T(x + bw - 41, y + 36, "GPU", 15, CARD, "middle", "bold", family=MONO))
        s.append(T(x + 22, y + 78, b1, 19, INK))
        s.append(T(x + 22, y + 104, b2, 19, INK))
    A = f'stroke="{MUTED}" stroke-width="2.2" marker-end="url(#ah)"'
    m0, m1 = Y(0) + bh / 2, Y(1) + bh / 2
    s.append(f'<line x1="{X(0) + bw + 6}" y1="{m0}" x2="{X(1) - 6}" y2="{m0}" {A}/>')
    s.append(f'<line x1="{X(1) + bw + 6}" y1="{m0}" x2="{X(2) - 6}" y2="{m0}" {A}/>')
    s.append(f'<line x1="{X(2) + bw / 2}" y1="{Y(0) + bh + 6}" x2="{X(2) + bw / 2}" y2="{Y(1) - 6}" {A}/>')
    s.append(f'<line x1="{X(2) - 6}" y1="{m1}" x2="{X(1) + bw + 6}" y2="{m1}" {A}/>')
    s.append(f'<line x1="{X(1) - 6}" y1="{m1}" x2="{X(0) + bw + 6}" y2="{m1}" {A}/>')
    s.append(f'<line x1="{X(0) + bw / 2}" y1="{Y(1) - 6}" x2="{X(0) + bw / 2}" y2="{Y(0) + bh + 6}" stroke="{GREY}" '
             f'stroke-width="2.2" stroke-dasharray="7 5" marker-end="url(#ah)"/>')
    s.append(T(X(0) + bw / 2 + 14, Y(0) + bh + gy / 2, "iterate", 17, MUTED, family=MONO))
    s.append("</svg>")
    body = (tiles([("Gradient steps", "0", "the author never backpropagates", False),
                   ("Tensors changed per edit", "9", "a neuron edit in three layers", False),
                   ("Author session", '90 <small>min</small>', "CPU only", False)])
            + f'<div class="chart" style="margin-top:26px">{"".join(s)}</div>'
            + '<div class="foot">Each edit is written with its own random control; the author never sees evaluation results before predicting.</div>')
    render(page("An LLM edits an agent's weights, no gradients", "Diagnose, locate, write, predict, verify, score. Only step 5 (orange) uses a GPU.", body),
           "fig1_loop")


# ---------- 2. exploration odds ratios ----------
def explore():
    import math
    bins = ["0–2", "3–4", "5–7", "8+"]; orr = [0.76, 1.42, 1.72, 0.12]
    lo = [0.39, 0.86, 0.97, 0.04]; hi = [1.48, 2.35, 3.06, 0.38]; share = [27, 34, 22, 17]
    H, L, R, Tp, B = 470, 92, SW - 20, 16, 380
    y = lambda v: Tp + (math.log(4.5) - math.log(v)) / (math.log(4.5) - math.log(0.03)) * (B - Tp)
    cw = (R - L) / 4
    xc = lambda i: L + cw * (i + 0.5)
    s = [f'<svg width="{SW}" height="{H}" viewBox="0 0 {SW} {H}">',
         f'<defs><marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{ORANGE}"/></marker></defs>']
    for i in range(4):
        if i % 2 == 0:
            s.append(f'<rect x="{L + cw * i}" y="{Tp}" width="{cw}" height="{B - Tp}" fill="{BAND}" opacity="0.6"/>')
    s.append(f'<rect x="{L + cw * 3}" y="{Tp}" width="{cw}" height="{B - Tp}" fill="{HOTBAND}" opacity="0.7"/>')
    for t in [0.05, 0.1, 0.25, 0.5, 1, 2, 4]:
        s.append(f'<line x1="{L - 6}" y1="{y(t):.1f}" x2="{L}" y2="{y(t):.1f}" stroke="{MUTED}" stroke-width="1.6"/>')
        s.append(T(L - 12, y(t), f"{t:g}", 17, MUTED, "end", family=MONO))
    s.append(f'<line x1="{L}" y1="{y(1):.1f}" x2="{R}" y2="{y(1):.1f}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(T(R - 8, y(1) - 13, "no effect", 17, MUTED, "end", family=MONO))
    s.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}" stroke-width="1.6"/>')
    s.append(f'<line x1="{L}" y1="{Tp}" x2="{L}" y2="{B}" stroke="{MUTED}" stroke-width="1.6"/>')
    s.append(T(L + 10, y(3.4), "↑ more likely solved", 17, MUTED, family=MONO))
    s.append(T(L + 10, y(0.04), "↓ less likely solved", 17, MUTED, family=MONO))
    for i in range(4):
        col = ORANGE if hi[i] < 1 else GREY
        x = xc(i)
        s.append(f'<line x1="{x}" y1="{y(lo[i]):.1f}" x2="{x}" y2="{y(hi[i]):.1f}" stroke="{col}" stroke-width="3"/>')
        for v in (lo[i], hi[i]):
            s.append(f'<line x1="{x - 9}" y1="{y(v):.1f}" x2="{x + 9}" y2="{y(v):.1f}" stroke="{col}" stroke-width="3"/>')
        s.append(f'<circle cx="{x}" cy="{y(orr[i]):.1f}" r="10" fill="{col}" stroke="{CARD}" stroke-width="2"/>')
        s.append(T(x + 20, y(orr[i]), f"{orr[i]:.2f}", 21, col, weight="bold" if col == ORANGE else "normal", family=SERIF))
        s.append(T(x, B + 24, bins[i], 20, HEAD, "middle", "bold", family=SERIF))
        s.append(T(x, B + 48, f"{share[i]}% of ep.", 17, MUTED, "middle", family=MONO))
    ax_, ay_ = xc(2) + 10, y(0.075)
    s.append(T(ax_ - 30, ay_ - 12, "8+ calls: odds of", 18, ORANGE, "middle", "bold"))
    s.append(T(ax_ - 30, ay_ + 12, "solving ~8× lower", 18, ORANGE, "middle", "bold"))
    s.append(f'<line x1="{ax_ + 62}" y1="{ay_ - 6}" x2="{xc(3) - 16}" y2="{y(0.12) + 4:.1f}" stroke="{ORANGE}" '
             f'stroke-width="1.8" marker-end="url(#ao)"/>')
    s.append(T((L + R) / 2, B + 80, "Tool calls before the first file edit", 18, INK, "middle"))
    s.append(T(18, (Tp + B) / 2, "Odds ratio (log)", 18, INK, "middle", extra=f'transform="rotate(-90 18 {(Tp + B) / 2})"'))
    s.append("</svg>")
    body = (tiles([("8+ calls before first edit", "17%", "of episodes", False),
                   ("Odds of solving", "0.12×", "95% CI 0.04–0.38", True),
                   ("Episodes", "2,210", "from 13 unedited runs", False)])
            + legend([("dot", GREY, "CI includes 1"), ("dot", ORANGE, "CI excludes 1")])
            + f'<div class="chart">{"".join(s)}</div>'
)
    render(page("Long exploration is where episodes are lost",
                "Odds of solving vs the same task's other episodes (95% CI)", body), "fig2_explore")


# ---------- 3. dose ----------
def dose():
    series = [("submit sooner", INDIGO, [0, 33, 93, 173], [(2, 0)]),
              ("think briefly", TEAL, [0, 10, 20, 39], [(2, 2)]),
              ("explore first", PLUM, [0, 8, 14, 26], [(2, 0), (2, 6), (2, 3), (3, 0)])]
    H, L, R, Tp, B = 450, 84, 780, 14, 370
    ymin, ymax = -26, 185
    y = lambda v: Tp + (ymax - v) / (ymax - ymin) * (B - Tp)
    x = lambda i: L + 40 + i * (R - L - 80) / 3
    s = [f'<svg width="{SW}" height="{H}" viewBox="0 0 {SW} {H}">']
    for i in range(4):
        if i % 2 == 1:
            bx0, bx1 = x(i) - (R - L - 80) / 6, min(R, x(i) + (R - L - 80) / 6)
            s.append(f'<rect x="{bx0}" y="{Tp}" width="{bx1 - bx0}" height="{B - Tp}" fill="{BAND}" opacity="0.6"/>')
    for t in [0, 50, 100, 150]:
        s.append(f'<line x1="{L}" y1="{y(t):.1f}" x2="{R}" y2="{y(t):.1f}" stroke="{GRID}" stroke-width="1.6"/>')
        s.append(T(L - 12, y(t), f"{t}", 17, MUTED, "end", family=MONO))
    s.append(f'<line x1="{L}" y1="{y(0):.1f}" x2="{R}" y2="{y(0):.1f}" stroke="{MUTED}" stroke-width="1.8"/>')
    s.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}" stroke-width="1.6"/>')
    # spread the five controls at mid horizontally so none hide each other
    off = {("submit sooner", 0): -30, ("think briefly", 0): -15, ("explore first", 0): 0, ("explore first", 1): 15,
           ("explore first", 2): 30, ("explore first", 3): 0}
    for name, col, ys, ctl in series:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3"/>')
        for i, v in enumerate(ys):
            s.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="8" fill="{col}" stroke="{CARD}" stroke-width="2"/>')
        dy = {39: 4, 26: 14}.get(ys[-1], 0)
        s.append(T(x(3) + 22, y(ys[-1]) + dy, f"{name}  +{ys[-1]}%", 21, col, weight="bold", family=SERIF))
    for name, col, ys, ctl in series:
        for j, (cx, c) in enumerate(ctl):
            xx = x(cx) + off[(name, j)]
            s.append(f'<rect x="{xx - 6:.1f}" y="{y(c) - 6:.1f}" width="12" height="12" fill="{CARD}" stroke="{ORANGE}" stroke-width="2.2"/>')
    s.append(T(x(2) + 30, y(-15), "random controls (□): 0 to 6%", 17, ORANGE, "start", "bold", family=MONO))
    for i, lab in enumerate(["unedited", "low", "mid", "high"]):
        s.append(T(x(i), B + 24, lab, 18, INK, "middle", family=MONO))
    s.append(T((L + R) / 2, B + 60, "Edit strength", 18, INK, "middle"))
    s.append(T(18, (Tp + B) / 2, "Toward targeted behaviour (%)", 18, INK, "middle",
               extra=f'transform="rotate(-90 18 {(Tp + B) / 2})"'))
    s.append("</svg>")
    body = (tiles([("Submit sooner", "+173%", "at high strength", False),
                   ("Random controls", "0 to 6%", "at matched strength", True)])
            + f'<div class="chart" style="margin-top:22px">{"".join(s)}</div>'
            + '<div class="foot">Controls at matched strength: submit, think at mid; explore first, three at mid, one at high.</div>')
    render(page("Edits move their decision with dose; controls don't",
                "Change in the targeted decision vs the unedited model, held-out replay", body), "fig3_dose")


# ---------- rows plots (4, 5, 6) ----------
def rows_svg(rows, xlim, xticks, xlabel, ref, refl, rh, labw=330, rule=None, link=False, bands=None, gaps=None,
             offscale_note=None, signed=False):
    n = len(rows)
    gaps = gaps or {}
    top = 40
    ys, acc = [], top
    for i in range(n):
        acc += gaps.get(i, 0)
        ys.append(acc + rh / 2)
        acc += rh
    B = acc + 6
    H = B + 74
    L, R = labw, SW - 80
    X = lambda v: L + (v - xlim[0]) / (xlim[1] - xlim[0]) * (R - L)
    s = [f'<svg width="{SW}" height="{H}" viewBox="0 0 {SW} {H}">']
    for i in range(n):
        if (i % 2 == 0) if bands is None else (i in bands):
            s.append(f'<rect x="0" y="{ys[i] - rh / 2}" width="{SW}" height="{rh}" fill="{BAND}" opacity="0.75"/>')
    if rule is not None:
        s.append(f'<rect x="{X(rule)}" y="{top - 6}" width="{R - X(rule)}" height="{B - top + 6}" fill="{ZONE}" opacity="0.8"/>')
        s.append(f'<line x1="{X(rule)}" y1="{top - 8}" x2="{X(rule)}" y2="{B}" stroke="{GREEN}" stroke-width="2"/>')
        s.append(T(X(rule) - 8, top - 22, "pass bar: mean ≥ +8", 15, GREEN, "end", "bold", family=MONO))
    for t in xticks:
        s.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{B}" stroke="{GRID}" stroke-width="1.6"/>')
        lab = minus(f"{t:+d}" if signed and t else str(t))
        s.append(T(X(t), B + 22, lab, 17, MUTED, "middle", family=MONO))
    s.append(f'<line x1="{X(ref):.1f}" y1="{top - 8}" x2="{X(ref):.1f}" y2="{B}" stroke="{MUTED}" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(T(X(ref), top - 22, refl, 17, MUTED, "middle", family=MONO))
    s.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="{MUTED}" stroke-width="1.6"/>')
    s.append(T((L + R) / 2, B + 56, xlabel, 18, INK, "middle"))
    for i, (lab, vals, kind) in enumerate(rows):
        col = {"e": INDIGO, "c": ORANGE, "n": GREY}[kind]
        yc = ys[i]
        m = sum(vals) / len(vals)
        lcol = {"e": HEAD, "c": ORANGE, "n": GREY}[kind]
        s.append(T(L - 18, yc, lab, 19, lcol, "end", "bold" if kind == "e" else "normal"))
        if link and len(vals) > 1:
            a, b = max(min(vals), xlim[0] + 0.2), max(max(vals), xlim[0] + 0.2)
            s.append(f'<line x1="{X(a):.1f}" y1="{yc}" x2="{X(b):.1f}" y2="{yc}" stroke="{col}" stroke-width="2" opacity="0.35"/>')
        seen = {}
        for v in vals:
            vv = max(v, xlim[0] + 0.2)
            k = seen.get(v, 0); seen[v] = k + 1
            dy = (k - (vals.count(v) - 1) / 2) * min(0.24 * rh, 11)
            if kind == "c":
                s.append(f'<rect x="{X(vv) - 6:.1f}" y="{yc + dy - 6:.1f}" width="12" height="12" fill="{CARD}" '
                         f'stroke="{col}" stroke-width="2.2" opacity="0.9"/>')
            else:
                s.append(f'<circle cx="{X(vv):.1f}" cy="{yc + dy:.1f}" r="6.5" fill="{col}" stroke="{CARD}" '
                         f'stroke-width="1.5" opacity="0.8"/>')
        mm = max(m, xlim[0] + 0.2)
        s.append(f'<rect x="{X(mm) - 2.5:.1f}" y="{yc - rh * 0.36:.1f}" width="5" height="{rh * 0.72:.1f}" fill="{col}"/>')
        val = minus(f"{m:+.1f}" if signed else f"{m:.1f}")
        s.append(T(R + 14, yc, val, 19, col, "start", "bold", family=MONO))
        if offscale_note and lab == offscale_note[0]:
            s.append(T(X(xlim[0] + 0.2) + 14, yc, offscale_note[1], 17, INDIGO, family=MONO))
    s.append("</svg>")
    return "".join(s)


def solve():
    rows = [("unedited (13 runs)", [28, 24, 22, 29, 22, 29, 27, 26, 25, 27, 30, 28, 25], "n"),
            ("random controls (14)", [27, 23, 26, 27, 27, 25, 28, 24, 23, 27, 25, 25, 25, 26], "c"),
            ("explore-more 0.125 (4)", [31, 30, 28, 33], "e"), ("explore-more 0.25 (7)", [32, 29, 30, 31, 29, 26, 27], "e"),
            ("explore-more 0.5 (10)", [28, 31, 33, 30, 26, 32, 30, 24, 28, 27], "e"), ("explore-more 1.0 (1)", [18], "e")]
    svg = rows_svg(rows, (16, 35), [16, 20, 24, 28, 32], "Tasks solved of 170", 26.3, "unedited mean", 58, labw=290)
    body = (tiles([("Unedited", "26.3", "mean of 13 runs", False),
                   ("Best dose 0.125", "30.5", "explore-more, 4 runs", False),
                   ("Too strong 1.0", "18", "explore-more, 1 run", True)])
            + legend([("dot", INDIGO, "edited run"), ("sq", ORANGE, "random control"), ("dot", GREY, "unedited run"),
                      ("bar", INK, "mean")])
            + f'<div class="chart">{svg}</div>'
)
    render(page("Low doses are ahead of unedited and controls",
                "Tasks solved of 170 per run (dot = run, bar = mean, (n) = runs)", body), "fig4_solve")


def arena():
    rows = [("Claude: first-edit gate", [9, 2, 3, 9, 5, 2], "e"), ("Claude: early-calls gate", [6, 2, 0, 9, 7, 4], "e"),
            ("Claude's random control", [1, 1, 4, 1], "c"),
            ("GPT: gate + rotation", [4, 0, 2, 8, 2, -2], "e"), ("GPT's random control", [-1, -3, 6, 0], "c"),
            ("Gemini: explore + concise", [7, 3, 7], "e"), ("Gemini's random control", [8, -3, 4], "c")]
    svg = rows_svg(rows, (-4, 10), [-4, 0, 4, 8], "Tasks gained vs unedited (of 170)", 0, "unedited", 56, labw=330,
                   bands={0, 1, 2, 5, 6}, gaps={3: 12, 5: 12}, signed=True)
    body = (tiles([("Claude: first-edit gate", "+5.0", "mean of 6 batches", False),
                   ("Claude's random control", "+1.8", "mean of 4 batches", True)])
            + legend([("dot", INDIGO, "edit, one batch"), ("sq", ORANGE, "random control, one batch"),
                      ("bar", INK, "mean")])
            + f'<div class="chart">{svg}</div>'
            + '<div class="foot">Each author\'s random control sits directly under its edits.</div>')
    render(page("Three LLM authors, one brief",
                "Solved tasks minus unedited, same batch (dot = batch, bar = mean)", body), "fig5_arena")


def round4():
    data = [("pe_L5", [5, 5]), ("pe_lo", [4, 6]), ("pe_L20", [1, 8]), ("u36", [1, 9]), ("u63", [3, 4]),
            ("late60", [6, 0]), ("pre_early", [3, 2]), ("late30", [2, 2]), ("pe_c6", [-1, 5]), ("u33_L5", [-2, 5]),
            ("pre15", [5, -2]), ("pe_hi", [-2, 3]), ("u33_c6", [-4, 5]), ("pre35", [3, -3]), ("pu", [3, -3]),
            ("u33", [-4, 2]), ("pe_L22", [1, -4]), ("pe_L14 (broke)", [-22, -28])]
    data = sorted(data[:-1], key=lambda d: -sum(d[1])) + data[-1:]
    rows = [(n, v, "e") for n, v in data] + [("control u33_ctrl", [7, 1], "c"), ("control pe_ctrl", [-4, 0], "c")]
    svg = rows_svg(rows, (-8, 12), [-8, -4, 0, 4, 8, 12], "Tasks gained vs unedited (of 170)", 0, "unedited", 36,
                   labw=250, rule=None, link=True, gaps={18: 10}, offscale_note=("pe_L14 (broke)", "◂ −22, −28 off scale"),
                   signed=True)
    body = (tiles([("Adapters", "20", "two batches each", False),
                   ("Best edit", "+5.0", "mean of two batches (3 edits)", True),
                   ("Best control", "+4.0", "u33_ctrl", False)])
            + legend([("dot", INDIGO, "edit, one batch"), ("sq", ORANGE, "random control"), ("bar", INK, "mean")])
            + f'<div class="chart">{svg}</div>'
)
    render(page("Round 4: 18 new edits, best mean +5 tasks", "Tasks gained vs the unedited model of the same batch; two batches per edit (bar = mean)", body), "fig6_round4")


if __name__ == "__main__":
    for theme in ("light", "dark"):
        set_theme(theme)
        loop(); explore(); dose(); solve(); arena(); round4()
    print("ok")
