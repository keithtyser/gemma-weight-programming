"""D1 summary: mean and 95% CI (normal, over decision points) of every per-point statistic in rows.jsonl.

Usage: python d1_summarize.py rows.jsonl meta.json out.json
Each decision point is one unit; no pooling across points before averaging (per-point pooled statistics are averaged).
"""
import json, math, sys
from collections import defaultdict


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if not n:
        return {"n": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h, "min": min(xs), "max": max(xs)}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    meta = json.load(open(sys.argv[2]))
    acc = defaultdict(list)
    for r in rows:
        for pair, p in r["pairs"].items():
            for cmp_, layers in p["resid"].items():
                for layer, st in layers.items():
                    for k, v in st.items():
                        acc[(pair, "resid", cmp_, layer, k)].append(v)
            for arm, layers in p["proj"].items():
                for layer, st in layers.items():
                    for k, v in st.items():
                        acc[(pair, "proj", arm, layer, k)].append(v)
            for cmp_, st in p["kl"].items():
                for k, v in st.items():
                    acc[(pair, "kl", cmp_, k)].append(v)
            for cmp_, v in p["top1"].items():
                acc[(pair, "top1", cmp_)].append(v)
    out = {"n_points": len(rows), "n_label1": sum(r["label"] for r in rows),
           "prompt_tokens_mean": sum(r["P"] for r in rows) / len(rows),
           "seq_tokens_mean": sum(r["L"] for r in rows) / len(rows), "meta": meta, "stats": {}}
    for key, xs in acc.items():
        d = out["stats"]
        for k in key[:-1]:
            d = d.setdefault(k, {})
        d[key[-1]] = ms(xs)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    S = out["stats"]
    f = lambda d: f"{d['mean']:.4f} [{d['lo']:.4f},{d['hi']:.4f}]" if d.get("n") else "-"
    print("points", len(rows))
    for pair in S:
        print("==", pair)
        for cmp_ in ("A|N", "A|V", "C|CN", "C|CV", "A|C", "N|V"):
            for layer in ("18", "22", "28", "41"):
                st = S[pair]["resid"][cmp_][layer]
                print(f" {cmp_:5s} L{layer} cos {f(st['cos_pooled'])} nr {f(st['norm_ratio_pooled'])} r2 {f(st['r2'])} "
                      f"cos_last {f(st['cos_last'])} nr_last {f(st['norm_ratio_last'])} r2_last {f(st['r2_last'])}")
        for arm, layers in S[pair]["proj"].items():
            print(f" proj {arm:3s} " + " ".join(f"L{l} all {st['all']['mean']:.3f} last {st['last']['mean']:.3f} comp {st['comp']['mean']:.3f};"
                                                 for l, st in layers.items()))
        for cmp_, st in S[pair]["kl"].items():
            print(f" KL {cmp_:8s} mean {st['mean']['mean']:.5f} [{st['mean']['lo']:.5f},{st['mean']['hi']:.5f}] first {st['first']['mean']:.6f}")
        print(" top1 " + " ".join(f"{k} {v['mean']:.4f};" for k, v in S[pair]["top1"].items()))


if __name__ == "__main__":
    main()
