"""D1 part 2 summary: read-outs of v2/steer/replay.py per arm, and paired agreement between arms.

Usage: python d1_regen_summarize.py OUT.json PAIR=DIR [PAIR=DIR ...] [--vllm NAME=replay_file.jsonl ...]
DIR holds rows_*.jsonl written by d1_regen.py. --vllm files are vLLM replay rows (replay_<adapter>.jsonl) joined on
(run_key, call) and compared with the HF arm NAME (engine cross-check).
"""
import glob, json, math, sys
from collections import Counter

TOOLS = (("edit", "edit_file"), ("run", "run_command"), ("read", "read_file"), ("submit", "submit_patch"), ("none", "none"))


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(m, 4), "lo": round(m - 1.96 * sd / math.sqrt(n), 4), "hi": round(m + 1.96 * sd / math.sqrt(n), 4), "n": n}


def readout(rs):
    cnt = Counter(r["tool"] for r in rs)
    eds = [r["old_ok"] for r in rs if r.get("old_ok") is not None]
    out = {"n": len(rs)}
    out.update({k: round(cnt[t] / len(rs), 3) for k, t in TOOLS})
    out.update({"think": round(sum(r["think"] for r in rs) / len(rs), 1), "tokens": round(sum(r["n_tokens"] for r in rs) / len(rs), 1),
                "old_ok": round(sum(eds) / len(eds), 3) if eds else None, "n_old": len(eds),
                "length": sum(r["finish"] == "length" for r in rs)})
    return out


def paired(rows, x, y):
    a, b = [r["arms"][x] for r in rows], [r["arms"][y] for r in rows]
    e = lambda r: float(r["tool"] == "edit_file")
    return {"n": len(rows), "same_tool": round(sum(p["tool"] == q["tool"] for p, q in zip(a, b)) / len(rows), 3),
            "edit_rate_diff": ci([e(p) - e(q) for p, q in zip(a, b)]),
            "think_diff": ci([p["think"] - q["think"] for p, q in zip(a, b)]),
            "abs_think_diff_mean": round(sum(abs(p["think"] - q["think"]) for p, q in zip(a, b)) / len(rows), 1)}


def main():
    out_path, args = sys.argv[1], sys.argv[2:]
    vllm, dirs = {}, {}
    i = 0
    while i < len(args):
        if args[i] == "--vllm":
            i += 1
            while i < len(args):
                k, v = args[i].split("=", 1)
                vllm[k] = {(r["run_key"], r["call"]): r for r in map(json.loads, open(v))}
                i += 1
        else:
            k, v = args[i].split("=", 1)
            dirs[k] = v
            i += 1
    res = {}
    for pair, d in dirs.items():
        rows = [json.loads(l) for f in sorted(glob.glob(d + "/rows_*.jsonl")) for l in open(f)]
        metas = [json.load(open(f)) for f in sorted(glob.glob(d + "/meta_*.json"))]
        arms = list(rows[0]["arms"])
        R = {"n_points": len(rows), "n_label1": sum(r["label"] for r in rows), "meta": metas,
             "sec_per_point": round(sum(r["sec"] for r in rows) / len(rows), 1), "readouts": {}, "paired": {},
             "kl_first": {k: ci([r["kl_first"][k] for r in rows]) for k in rows[0]["kl_first"]},
             "identical_tokens_to_A": {n: round(sum(r["arms"][n]["same_as_A"] for r in rows) / len(rows), 3) for n in arms},
             "common_prefix_with_A_mean": {n: round(sum(r["arms"][n]["common_prefix_A"] for r in rows) / len(rows), 1) for n in arms}}
        for n in arms:
            R["readouts"][n] = {str(lab): readout([r["arms"][n] for r in rows if lab == "all" or r["label"] == lab])
                                for lab in (1, 0, "all")}
        for x, y in (("N", "A"), ("V", "A"), ("base", "A"), ("C", "A"), ("CV", "C"), ("base", "C"), ("A", "base"), ("N", "base"),
                     ("V", "base"), ("C", "base"), ("CV", "base"), ("N", "V")):
            R["paired"][f"{x}|{y}"] = paired(rows, x, y)
        R["vllm"] = {}
        for spec, tab in vllm.items():
            p_, arm, name = spec.split(":")  # PAIR:ARM:vllm adapter name
            if p_ != pair:
                continue
            j = [(r["arms"][arm], tab[(r["run_key"], r["call"])]) for r in rows if (r["run_key"], r["call"]) in tab]
            R["vllm"][f"{arm}~{name}"] = {"n": len(j), "same_tool": round(sum(h["tool"] == v["tool"] for h, v in j) / len(j), 3),
                                          "hf": readout([h for h, _ in j]), "vllm": readout([{**v, "n_tokens": v["n_tokens"]} for _, v in j])}
        res[pair] = R
        print("==", pair, R["n_points"], "points", R["sec_per_point"], "s/point")
        for n in arms:
            print(" ", n.ljust(5), json.dumps(R["readouts"][n]["all"]))
        for k, v in R["paired"].items():
            print("  paired", k.ljust(8), "same_tool", v["same_tool"], "d_edit", v["edit_rate_diff"]["mean"],
                  [v["edit_rate_diff"]["lo"], v["edit_rate_diff"]["hi"]], "d_think", v["think_diff"]["mean"],
                  [v["think_diff"]["lo"], v["think_diff"]["hi"]])
        print("  identical to A", R["identical_tokens_to_A"], "prefix", R["common_prefix_with_A_mean"])
        for k, v in R["vllm"].items():
            print("  vllm", k, v["n"], "same_tool", v["same_tool"], "hf", json.dumps(v["hf"]), "vllm", json.dumps(v["vllm"]))
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
