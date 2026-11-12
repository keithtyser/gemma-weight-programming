"""For failed edits whose old_string is verbatim in an earlier tool result: was that text read before the file was edited (stale)? Run from the repo root. Usage: python -m v2.steer.forensics.edit_fail_stale_text b3_base,b4_base

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: CPU-only trace forensics for the weight-edit paper. Reads $STEER_EVAL/<arm>/**/task_results.jsonl and
  trace_<task>.json (default results/steer_eval). Prints tables; writes nothing.
"""
import os, sys, json, collections, difflib
from pathlib import Path
from v2.steer.editfail import classify
root = Path(os.environ.get("STEER_EVAL", "results/steer_eval"))
src = collections.Counter(); cls_first = collections.Counter(); shown = collections.Counter()
for arm in sys.argv[1].split(","):
    for f in (root / arm).rglob("trace_*.json"):
        seen = ""; parts = []; k = 0; edited = set()
        for s in json.loads(f.read_text())["steps"]:
            tcs = s.get("tool_calls") or []
            obs = (s.get("observation") or {}).get("content") or ""
            if not tcs: continue
            name = tcs[0].get("function_name"); a = tcs[0].get("arguments") or {}
            fp = str(a.get("filepath") or a.get("path") or "")
            if name == "edit_file":
                old = str(a.get("old_string") or ""); bad = '"status": "error"' in obs[:300]
                if bad:
                    c, why = classify(old, seen)
                    cls_first[(c, "first" if k == 0 else "later")] += 1
                    if c == "verbatim":
                        w = [(n, p) for n, p, t in parts if old in t]
                        kind = "same file read" if any(n == "read_file" and p == fp for n, p in w) else "other file read" if any(n == "read_file" for n, p in w) else "command output" if w else "spans / new_string"
                        src[(kind, "file already edited" if fp in edited else "file not yet edited", "first" if k == 0 else "later")] += 1
                        if kind == "same file read" and fp not in edited and shown["v"] < 5:
                            shown["v"] += 1; print("VERB", arm, f.stem[6:], fp, "len", len(old), repr(old[:160]), "| ERR", obs[60:330].replace(chr(10), " "))
                    if c == "near" and k == 0 and shown["n"] < 5:
                        shown["n"] += 1; print("NEAR", f.stem[6:], "len", len(old), why[:300])
                else:
                    edited.add(fp); seen += "\n" + str(a.get("new_string") or "")
                k += 1
            try:
                o = json.loads(obs); t = str(o.get("content") or "") + str(o.get("stdout") or "")
            except (ValueError, AttributeError): t = obs
            seen += "\n" + t; parts.append((name, fp, t))
print("class x first/later:", sorted(cls_first.items(), key=lambda x: -x[1]))
print("verbatim source:"); [print("  ", k, v) for k, v in src.most_common()]
