"""Check a served arm: tool call + reasoning parse, and first-token log-probabilities at temperature 0.

    python3 serve_check.py PORT arm...
"""
import json, sys, urllib.request

port, arms = sys.argv[1], sys.argv[2:]
tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]}}}
    for n, d in [("bash", "Run a shell command; arg is the command."),
                 ("edit_file", "Edit a file; arg is the path."), ("submit", "Submit the work; arg is a note.")]]
msgs = [{"role": "system", "content": "You are a coding agent working in /repo. Use the tools."},
        {"role": "user", "content": "The test tests/test_parse.py::test_empty fails with KeyError 'name'. Fix the bug."}]
for a in arms:
    body = dict(model=a, messages=msgs, tools=tools, temperature=0, max_tokens=300, logprobs=True, top_logprobs=5, seed=0)
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=120))
    c = d["choices"][0]; m = c["message"]
    tc = [(t["function"]["name"], t["function"]["arguments"][:60]) for t in (m.get("tool_calls") or [])]
    reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    lp = (c.get("logprobs") or {}).get("content") or []
    first = [(x["token"], round(x["logprob"], 3)) for x in lp[0]["top_logprobs"][:4]] if lp else None
    print(a, "| finish", c["finish_reason"], "| reasoning chars", len(reasoning), "| content", repr((m.get("content") or "")[:60]),
          "| tools", tc, "| ntok", d["usage"]["completion_tokens"], "| prompt", d["usage"]["prompt_tokens"], "| first", first)
