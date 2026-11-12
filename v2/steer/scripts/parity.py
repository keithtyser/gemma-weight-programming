"""Greedy parity between two servers. Usage: parity.py URL_A URL_B arm [arm ...]. Prints text match and top-logprob gap."""
import json, sys, urllib.request

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number, then explain its complexity in one sentence.",
    "A test fails with KeyError: 'name' in parse_config. List three likely causes.",
]


def ask(url, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 48,
            "logprobs": True, "top_logprobs": 1}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    c = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]
    lp = [(t["token"], t["logprob"]) for t in (c.get("logprobs") or {}).get("content") or []]
    return (c["message"].get("content") or "") + "|" + (c["message"].get("reasoning_content") or ""), lp


a, b = sys.argv[1], sys.argv[2]
for arm in sys.argv[3:]:
    for p in PROMPTS:
        ta, la = ask(a, arm, p)
        tb, lb = ask(b, arm, p)
        n = 0
        while n < min(len(la), len(lb)) and la[n][0] == lb[n][0]:
            n += 1
        gap = max((abs(la[i][1] - lb[i][1]) for i in range(n)), default=0.0)
        print(arm, "same_text", ta == tb, "tokens", len(la), len(lb), "common_prefix", n, "max_logprob_gap", round(gap, 4))
