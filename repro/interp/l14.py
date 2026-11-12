"""Read-only: how episodes end, per arm (last tool, steps without tool calls, malformed calls)."""
import glob, json, sys, collections
ROOT = "/path/to/gemma-v2/results/steer_eval"
for arm in sys.argv[1:]:
    last = collections.Counter(); notool = 0; steps = 0; eps = 0; badargs = 0; calls = 0
    for f in glob.glob(f"{ROOT}/{arm}/*/shard*/traces/trace_*.json"):
        try: st = json.load(open(f))["steps"]
        except Exception: continue
        eps += 1; names = []
        for s in st:
            if s.get("source") not in (None, "agent") and "tool_calls" not in s: continue
            tcs = s.get("tool_calls") or []
            steps += 1; notool += (len(tcs) == 0)
            for tc in tcs:
                names.append(tc.get("function_name")); calls += 1
                a = tc.get("arguments")
                if not isinstance(a, dict) or not a: badargs += 1
        last[names[-1] if names else "<none>"] += 1
    print(arm, "eps", eps, "calls/ep", round(calls/max(eps,1), 2), "steps_without_tool_call", round(notool/max(steps,1), 3),
          "empty_or_nondict_args", round(badargs/max(calls,1), 3), "last_tool", dict(last.most_common(4)))
