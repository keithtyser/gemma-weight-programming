"""Why do edit_file calls fail? Classify every failed edit of the given eval arms against what the agent had read (CPU).

Author: Claude Fable 5 (keef)
Date: 11-November-2026
PURPOSE: Diagnose step for the "copy fidelity" failure. For each edit_file call whose result is an error, compare
  old_string with the text the agent had seen so far in the episode (read_file contents and command output):
      verbatim      old_string is in the seen text (the file changed since, or the read was of another file)
      whitespace    equal after removing all whitespace (indentation / blank-line error)
      near          a seen window of the same length has difflib ratio >= 0.9 (a few wrong characters)
      elision       old_string contains a placeholder such as "..." or "rest of the code"
      partial       the first line is in the seen text, the rest is not close
      unseen        the first non-empty line is not in the seen text at all
  Prints counts, the share by class, and --show examples per class with the differing characters.
  Usage: python -m v2.steer.editfail --root results/steer_eval --arms b3_base,b4_base [--show 3]
SRP/DRY check: Pass - v2.steer.tracefeat counts failed edits per episode; this explains them.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
from pathlib import Path

WS = re.compile(r"\s+")


def classify(old: str, seen: str) -> tuple[str, str]:
    if not old.strip():
        return "empty", ""
    if old in seen:
        return "verbatim", ""
    if WS.sub("", old) in WS.sub("", seen):
        return "whitespace", ""
    if "..." in old or "rest of the" in old or "remains the same" in old:
        return "elision", ""
    lines = [ln for ln in old.split("\n") if ln.strip()]
    first = lines[0].strip()
    at = seen.find(first)
    if at < 0:
        return "unseen", first[:120]
    best, bw = 0.0, ""
    while at >= 0 and best < 0.9:
        start = seen.rfind("\n", 0, at) + 1
        w = seen[start:start + len(old) + 40]
        r = difflib.SequenceMatcher(None, old, w[:len(old)], autojunk=False).ratio()
        if r > best:
            best, bw = r, w[:len(old)]
        at = seen.find(first, at + 1)
    if best >= 0.9:
        ops = [f"{bw[i1:i2]!r}->{old[j1:j2]!r}" for tag, i1, i2, j1, j2 in
               difflib.SequenceMatcher(None, bw, old, autojunk=False).get_opcodes() if tag != "equal"]
        return "near", "; ".join(ops[:4])[:240]
    return "partial", first[:120]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m v2.steer.editfail")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--show", type=int, default=3)
    a = p.parse_args(argv)
    cnt: collections.Counter[str] = collections.Counter()
    ex: dict[str, list[str]] = collections.defaultdict(list)
    n_edit = 0
    for arm in a.arms.split(","):
        for f in (a.root / arm).rglob("trace_*.json"):
            seen = ""
            for s in json.loads(f.read_text())["steps"]:
                tcs = s.get("tool_calls") or []
                obs = (s.get("observation") or {}).get("content") or ""
                if not tcs:
                    continue
                name = tcs[0].get("function_name")
                args = tcs[0].get("arguments") or {}
                if name == "edit_file":
                    n_edit += 1
                    if '"status": "error"' in obs[:300]:
                        c, why = classify(str(args.get("old_string") or ""), seen)
                        cnt[c] += 1
                        if len(ex[c]) < a.show and why:
                            ex[c].append(f"{f.stem[6:]}: {why}")
                    else:
                        seen += "\n" + str(args.get("new_string") or "")
                try:
                    o = json.loads(obs)
                    seen += "\n" + str(o.get("content") or "") + str(o.get("stdout") or "")
                except (ValueError, AttributeError):
                    seen += "\n" + obs
    tot = sum(cnt.values())
    print(f"edit_file calls {n_edit} failed {tot} ({tot / n_edit:.2f})")
    for c, k in cnt.most_common():
        print(f"{c:11s} {k:4d} {k / tot:.2f}")
        for e in ex[c]:
            print("     ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
