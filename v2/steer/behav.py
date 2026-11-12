"""Behaviour features per eval run from ATIF traces; resolved vs failed contrast. Usage: behav.py ARM_DIR [ARM_DIR...]"""
import collections, glob, json, os, statistics as st, sys

def feats(tr):
    calls = []
    for s in tr["steps"]:
        for tc in s.get("tool_calls") or []:
            obs = s.get("observation")
            calls.append((tc["function_name"], json.dumps(tc.get("arguments"), sort_keys=True), json.dumps(obs)[:4000] if obs else ""))
    names = [c[0] for c in calls]
    edits = [i for i, n in enumerate(names) if n in ("edit_file", "write_file", "create_file", "apply_patch")]
    def is_test(c):
        return c[0] == "run_command" and ("pytest" in c[1] or "unittest" in c[1] or "tox" in c[1])
    tests = [i for i, c in enumerate(calls) if is_test(c)]
    last_edit = edits[-1] if edits else -1
    files = {json.loads(calls[i][1]).get("filepath") for i in edits}
    rep = collections.Counter((c[0], c[1]) for c in calls)
    nudges = sum(1 for s in tr["steps"] if (s.get("extra") or {}).get("event_type") == "continuation_nudge")
    err = sum(1 for c in calls if '\\"status\\": \\"error\\"' in c[2] or '"status": "error"' in c[2])
    return {
        "calls": len(calls), "edits": len(edits), "files_edited": len(files), "tests_run": len(tests),
        "test_before_edit": int(any(i < (edits[0] if edits else 10**9) for i in tests)),
        "test_after_last_edit": int(any(i > last_edit for i in tests)) if edits else 0,
        "any_cmd_after_last_edit_py": int(any(i > last_edit and c[0] == "run_command" and "python" in c[1] for i, c in enumerate(calls))) if edits else 0,
        "submit": int("submit_patch" in names), "nudges": nudges, "max_repeat": max(rep.values()) if rep else 0,
        "repeat3": int(any(v >= 3 for v in rep.values())), "err_frac": err / max(len(calls), 1),
        "first_edit_idx": edits[0] if edits else -1, "edit_fail": sum(1 for i in edits if "error" in calls[i][2][:200]),
        "comp_tok": int(tr["final_metrics"]["total_completion_tokens"]),
        "edits_test_files": int(any("test" in (f or "") for f in files)),
    }

for arm in sys.argv[1:]:
    res = {}
    for f in glob.glob(arm + "/*/*/task_results.jsonl"):
        for l in open(f):
            r = json.loads(l); res[r["instance_id"]] = r
    rows = []
    for f in glob.glob(arm + "/*/*/traces/trace_*.json"):
        iid = os.path.basename(f)[6:-5]
        if iid not in res: continue
        r = res[iid]; e = r.get("error") or r.get("agent_error") or ""
        kind = "resolved" if r["resolved"] else "budget" if ("budget" in e or "timeout" in e) else "wrong"
        try: rows.append((kind, feats(json.load(open(f)))))
        except Exception as ex: print("skip", iid, ex)
    print("==", arm, collections.Counter(k for k, _ in rows))
    keys = list(rows[0][1])
    print("%-28s" % "feature" + "".join("%10s" % k for k in ("resolved", "wrong", "budget")))
    for k in keys:
        print("%-28s" % k + "".join("%10.2f" % (st.mean([f[k] for kk, f in rows if kk == g]) if any(kk == g for kk, _ in rows) else float("nan")) for g in ("resolved", "wrong", "budget")))
    names = collections.Counter()
