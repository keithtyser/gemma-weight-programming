"""Read-only: calls before first file edit, per arm per batch, from steer_eval traces."""
import glob, json, sys, os, statistics as st
ROOT = "/path/to/gemma-v2/results/steer_eval"
EDITS = ("edit_file", "write_file")
def arm_stats(d):
    res = {}
    for f in glob.glob(f"{d}/*/shard*/task_results.jsonl"):
        for line in open(f):
            try: r = json.loads(line)
            except Exception: continue
            res[r["instance_id"].lower()] = bool(r.get("resolved"))
    out = {}
    for f in glob.glob(f"{d}/*/shard*/traces/trace_*.json"):
        iid = f.split("/trace_")[-1][:-5].lower()
        if iid not in res: continue
        try: steps = json.load(open(f))["steps"]
        except Exception: continue
        names = [tc.get("function_name") for s in steps for tc in (s.get("tool_calls") or [])]
        fe = next((i for i, n in enumerate(names) if n in EDITS), None)
        out[iid] = (fe if fe is not None else len(names), fe is None, len(names), res[iid])
    return out
batches = sys.argv[1:]
rows = {}
for b in batches:
    arms = sorted(os.path.basename(p)[len(b)+1:] for p in glob.glob(f"{ROOT}/{b}_*"))
    if "base" not in arms: continue
    S = {a: arm_stats(f"{ROOT}/{b}_{a}") for a in arms}
    common = set(S["base"])
    for a in arms: common &= set(S[a])
    for a in arms:
        v = [S[a][i] for i in common]
        if not v: continue
        cb = [x[0] for x in v]
        rows[f"{b}_{a}"] = dict(batch=b, arm=a, n=len(v), med=st.median(cb), mean=round(sum(cb)/len(cb), 3),
            share8=round(sum(c >= 8 for c in cb)/len(cb), 4), share57=round(sum(5 <= c <= 7 for c in cb)/len(cb), 4),
            never=round(sum(x[1] for x in v)/len(v), 4), calls=round(sum(x[2] for x in v)/len(v), 2),
            solved=sum(x[3] for x in v))
for k, r in rows.items():
    print(json.dumps(r))
