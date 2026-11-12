"""D1 prep (CPU, evalhost): the 320 held-out replay decision points of v2.steer.replay (states_w5d, tag edit@0), one row each."""
import json, sys
sys.path.insert(0, "/path/to/gemma-v2/tii-steer")
from v2.steer.replay import read_prompt
from v2.steer.collect import read_call
S = "/path/to/gemma-v2/results/steer/"
rows = [r for r in map(json.loads, open(S + "states_w5d/states.jsonl")) if r["split"] == "heldout"]
n = 0
with open(sys.argv[1], "w") as fh:
    for r in rows:
        q = next((x for x in r["positions"] if x["tag"] == "edit@0"), None)
        if q is None:
            continue
        prompt, seen = read_prompt(r["request_log_path"], r["call"])
        if len(prompt) + 3072 > 32768:
            continue
        fh.write(json.dumps({"run_key": r["run_key"], "call": r["call"], "label": q["label"], "idx": q["idx"],
                             "abs": q["abs"], "prompt": prompt, "seen": seen,
                             "completion": read_call(r["request_log_path"], r["call"])["completion_token_ids"]}) + "\n")
        n += 1
print(n)
