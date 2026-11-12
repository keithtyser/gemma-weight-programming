"""M4: classification of zero-edit episodes and the 8+ MH OR restricted to episodes with >=1 file edit.
Reuses load/mh/strata_for from diag_recompute/recompute.py (same method as the published 0.12, 0.04-0.38).
Run: python -I m4_8plus_edited_only.py diag_recompute
"""
import importlib.util, os, sys, json
BASE = sys.argv[1]
spec = importlib.util.spec_from_file_location("rc", os.path.join(BASE, "recompute.py"))
rc = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0], BASE]; spec.loader.exec_module(rc)
eps = rc.load()
EDIT = ("edit_file", "write_file")
def npre(e):
    names = [c[0] for c in e["calls"]]
    return next((i for i, n in enumerate(names) if n in EDIT), len(names))
def edited(e): return any(c[0] in EDIT for c in e["calls"])
never = [e for e in eps if not edited(e)]
print("episodes", len(eps), "never edited", len(never))
for e in never:
    print("  never-edit: batch", e["batch"], e["task"], "calls", len(e["calls"]), "n_pre", npre(e), "solved", e["solved"])
for label, pool in (("all episodes", eps), (">=1 file edit", [e for e in eps if edited(e)])):
    for lo, hi, lab in ((0, 2, "0-2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 10**9, "8+")):
        inb = lambda e, lo=lo, hi=hi: lo <= npre(e) <= hi
        w = [e for e in pool if inb(e)]; wo = [e for e in pool if not inb(e)]
        r = rc.mh(rc.strata_for(pool, inb))
        print(f"{label:14s} bin {lab:4s} n={len(w):4d}/{len(pool)} solved_in={sum(e['solved'] for e in w)} "
              f"solved_out={sum(e['solved'] for e in wo)} OR={r['or']:.3f} CI=({r['lo']:.3f},{r['hi']:.3f}) p={r['cmh_p']:.2g}")
