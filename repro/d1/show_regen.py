import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    print(r["run_key"], r["call"], r["label"], r["P"], r["sec"], {k: round(v, 6) for k, v in r["kl_first"].items()})
    for n, x in r["arms"].items():
        print("  ", n, x["tool"], x["n_tokens"], x["think"], x["finish"], x["same_as_A"], x["common_prefix_A"])
