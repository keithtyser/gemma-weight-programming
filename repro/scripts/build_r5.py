"""Round-5 edits by author `claude`: dose and timing ladder of the `late` (commit after long exploration) neuron.

    python build_r5.py --edit NAME        writes ../edits/NAME/
Same closed-form construction as build.py / build_r3.py. Motivation and predictions: ../../round5_predictions.md.
"""
import json, sys
from pathlib import Path
import build
from build_r3 import late, L3, L5

build.EDITS.update({
    "late90":      dict(layers=L3, neurons=[late(0.9)]),
    "late60_x3":   dict(layers=L3, neurons=[late(0.6)]),            # write size 3x the solved late60 size per layer
    "late60_x5":   dict(layers=L3, neurons=[late(0.6)]),            # 5x; past the point where the class-gap target saturates
    "late60_c6":   dict(layers=L3, neurons=[late(0.6, 3, 6)]),      # commit push from call 6 instead of 8
    "late100_c6":  dict(layers=L3, neurons=[late(1.0, 3, 6)]),
    "late60_ctrl": dict(layers=L3, neurons=[late(0.6)], control=True),   # random write direction, matched to late60
})
CTRL_OF = {"late60_ctrl": "late60"}
CMUL = {"late60_x3": ("late60", 3.0), "late60_x5": ("late60", 5.0)}

if __name__ == "__main__":
    name = sys.argv[sys.argv.index("--edit") + 1]
    if name in CTRL_OF:
        src = Path(__file__).resolve().parent.parent / "edits" / CTRL_OF[name] / "report.json"
        R = json.loads(src.read_text())
        build.EDITS[name]["neurons"] = [dict(ns, _c={int(l): v["neurons"][ni]["c"] for l, v in R["layers"].items()})
                                        for ni, ns in enumerate(build.EDITS[name]["neurons"])]
    if name in CMUL:
        ref, m = CMUL[name]
        R = json.loads((Path(__file__).resolve().parent.parent / "edits" / ref / "report.json").read_text())
        build.EDITS[name]["neurons"] = [dict(ns, _cfix={int(l): m * v["neurons"][ni]["c"] for l, v in R["layers"].items()})
                                        for ni, ns in enumerate(build.EDITS[name]["neurons"])]
    build.main()
