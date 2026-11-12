"""Diagnosis from logged eval traces: what happens after a failed edit_file, and how exploration relates to solving.

Usage: python diag.py ARM_SUFFIX [ARM_SUFFIX ...]   (e.g. base bneg_t0.5); reads results/steer_eval/b{3..15}_<suffix>.
"""
import collections, glob, json, sys

ROOT = "diag_recompute/raw"


def episodes(arm):
    res = {}
    for f in glob.glob(f"{ROOT}/{arm}/*/shard*/task_results.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            res[r["instance_id"].lower()] = bool(r.get("resolved"))
    for f in glob.glob(f"{ROOT}/{arm}/*/shard*/traces/trace_*.json"):
        iid = f.replace(chr(92),"/").split("/trace_")[-1][:-5].lower()
        if iid not in res:
            continue
        calls = []
        for s in json.load(open(f, encoding="utf-8"))["steps"]:
            tcs = s.get("tool_calls") or []
            obs = json.dumps(s.get("observation") or {})
            for tc in tcs:
                name = tc.get("function_name")
                args = tc.get("arguments") or {}
                err = '\\"status\\": \\"error\\"' in obs or '"status": "error"' in obs
                calls.append((name, args.get("filepath") or args.get("path"), err if len(tcs) == 1 else None))
        yield iid, res[iid], calls


def main():
    for suf in sys.argv[1:]:
        C = collections.Counter()
        nxt = collections.Counter()
        retry = collections.Counter()
        ep = collections.defaultdict(lambda: [0, 0])
        pre = []
        for b in range(3, 16):
            for iid, ok, calls in episodes(f"b{b}_{suf}"):
                C["episodes"] += 1
                C["resolved"] += ok
                first_edit = next((i for i, c in enumerate(calls) if c[0] in ("edit_file", "write_file")), None)
                pre.append(first_edit if first_edit is not None else -1)
                first_fail_kind = None
                for i, (name, fp, err) in enumerate(calls):
                    if name != "edit_file":
                        continue
                    C["edit_calls"] += 1
                    if not err:
                        continue
                    C["edit_fail"] += 1
                    if i + 1 >= len(calls):
                        nxt["(end)"] += 1
                        continue
                    n = calls[i + 1][0]
                    kind = n if n in ("edit_file", "read_file") else "other"
                    nxt[kind] += 1
                    if first_fail_kind is None:
                        first_fail_kind = kind
                    # outcome of the next edit_file on this file, by whether the file was read in between
                    read_between = False
                    for name2, fp2, err2 in calls[i + 1:]:
                        if name2 == "read_file" and fp2 == fp:
                            read_between = True
                        if name2 == "edit_file" and fp2 == fp:
                            retry[(read_between, bool(err2))] += 1
                            break
                key = first_fail_kind or "no_failed_edit"
                ep[key][0] += ok
                ep[key][1] += 1
        if not C["episodes"]:
            continue
        pre_ok = [p for p in pre if p >= 0]
        print(f"== {suf}: episodes {C['episodes']} solved {C['resolved']} ({C['resolved'] / C['episodes']:.3f}); "
              f"edit_file calls {C['edit_calls']} failed {C['edit_fail']} ({C['edit_fail'] / max(1, C['edit_calls']):.3f}); "
              f"mean calls before first edit {sum(pre_ok) / max(1, len(pre_ok)):.2f}; never edited {pre.count(-1)}")
        tot = sum(nxt.values())
        print("   call right after a failed edit:", {k: f"{v} ({v / tot:.2f})" for k, v in nxt.most_common()})
        for rb in (False, True):
            n = retry[(rb, True)] + retry[(rb, False)]
            print(f"   next edit of the same file, file re-read in between={rb}: n={n} fails {retry[(rb, True)] / max(1, n):.3f}")
        print("   episode solve rate by the call after its FIRST failed edit:",
              {k: f"{v[0]}/{v[1]} ({v[0] / max(1, v[1]):.3f})" for k, v in ep.items()})


if __name__ == "__main__":
    main()
