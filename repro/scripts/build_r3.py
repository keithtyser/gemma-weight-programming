"""Round-3 wide-screen edits by author `claude` (same closed-form construction as build.py; one command per edit).

    python build_r3.py --edit NAME        writes ../edits/NAME/
Diagnosis used (13 unedited runs, 2,210 episodes): solving vs calls before the first edit is an inverted U
(0-2 calls OR 0.76, 3-4 calls OR 1.42, 5-7 calls OR 1.72, 8+ calls OR 0.12). Earlier edits only pushed "explore more".
New here: `late` pushes TOWARDS the first edit once the episode has made 8+ calls without editing; `u*` edits combine
the early explore push with the late commit push. The rest are dose / depth / gate-threshold variants of pre_early.
"""
import json, sys
from pathlib import Path
import build

PRE_ON, PRE_OFF = dict(tags=build.PRE + ["bg@prompt"]), dict(tags=build.POST)
POST = dict(tags=build.POST)


def pre(t):
    return dict(write=("edit@0", -1.0), target=t, on=PRE_ON, off=PRE_OFF)


def early(t, lo=4, hi=8):
    return dict(write=("edit@0", -1.0), target=t, on=dict(tags=build.PRE, call_max=lo),
                off=[dict(tags=build.PRE, call_min=hi), POST])


def late(t, lo=4, hi=8):
    return dict(write=("edit@0", 1.0), target=t, on=dict(tags=build.PRE, call_min=hi),
                off=[dict(tags=build.PRE, call_max=lo), POST])


L3, L5 = [18, 20, 22], [16, 18, 20, 22, 24]
build.EDITS.update({
    # inverted-U edits: explore more early, commit once exploration has run long
    "u33":      dict(layers=L3, neurons=[early(0.3), late(0.3)]),
    "u36":      dict(layers=L3, neurons=[early(0.3), late(0.6)]),
    "u63":      dict(layers=L3, neurons=[early(0.6), late(0.3)]),
    "u33_L5":   dict(layers=L5, neurons=[early(0.3), late(0.3)]),
    "u33_c6":   dict(layers=L3, neurons=[early(0.3, 4, 8), late(0.3, 6, 10)]),
    "late30":   dict(layers=L3, neurons=[late(0.3)]),
    "late60":   dict(layers=L3, neurons=[late(0.6)]),
    "pu":       dict(layers=L3, neurons=[pre(0.25), early(0.3), late(0.6)]),
    # pre_early (before-first-edit gate + early-calls gate) variants
    "pe_lo":    dict(layers=L3, neurons=[pre(0.15), early(0.2)]),
    "pe_hi":    dict(layers=L3, neurons=[pre(0.35), early(0.4)]),
    "pe_L5":    dict(layers=L5, neurons=[pre(0.25), early(0.3)]),
    "pe_L14":   dict(layers=[14, 16, 18], neurons=[pre(0.25), early(0.3)]),
    "pe_L20":   dict(layers=[20, 22, 24], neurons=[pre(0.25), early(0.3)]),
    "pe_L22":   dict(layers=[22, 24, 26], neurons=[pre(0.25), early(0.3)]),
    "pe_c6":    dict(layers=L3, neurons=[pre(0.25), early(0.3, 6, 10)]),
    "pre15":    dict(layers=L3, neurons=[pre(0.15)]),
    "pre35":    dict(layers=L3, neurons=[pre(0.35)]),
    # matched random-direction controls (same neurons, gates and |down|; built after the edit they match)
    "u33_ctrl": dict(layers=L3, neurons=[early(0.3), late(0.3)], control=True),
    "pe_ctrl":  dict(layers=L3, neurons=[pre(0.25), early(0.3)], control=True),
})
CTRL_OF = {"u33_ctrl": "u33", "pe_ctrl": "pre_early"}

if __name__ == "__main__":
    name = sys.argv[sys.argv.index("--edit") + 1]
    if name in CTRL_OF:
        src = Path(__file__).resolve().parent.parent / "edits" / CTRL_OF[name] / "report.json"
        R = json.loads(src.read_text())
        build.EDITS[name]["neurons"] = [dict(ns, _c={int(l): v["neurons"][ni]["c"] for l, v in R["layers"].items()})
                                        for ni, ns in enumerate(build.EDITS[name]["neurons"])]
    build.main()
