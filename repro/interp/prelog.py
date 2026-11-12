"""Read-only: pre-edit summaries from claude build logs (work/r3_*.log, r5_*.log)."""
import glob, json, os, re
W = "/path/to/gemma-v2/wsp_arena/claude/work"
for f in sorted(glob.glob(f"{W}/r3_*.log") + glob.glob(f"{W}/r5_*.log")):
    name = os.path.basename(f)[3:-4]
    layers = []
    other = []
    for line in open(f):
        m = re.match(r"^(\d+) (\{.*\})\s*$", line)
        if not m:
            if line.strip() and len(line) < 300: other.append(line.strip()[:200])
            continue
        L = int(m.group(1)); d = json.loads(m.group(2))
        v = d.get("verified_bf16_shift_over_gap", {})
        rel = v.get("rel_change_norm", {})
        along = {k: x for k, x in v.items() if k.startswith("along_")}
        cs = [n.get("c") for n in d.get("neurons", [])]
        writes = [n.get("write") for n in d.get("neurons", [])]
        layers.append(dict(L=L, c=[round(c, 3) if c is not None else None for c in cs], write=writes,
                           rel_max=round(max(rel.values()), 4) if rel else None,
                           rel_mean=round(sum(rel.values()) / len(rel), 4) if rel else None,
                           along={k: {g: x[g] for g in list(x)[:6]} for k, x in along.items()}))
    print(json.dumps(dict(edit=name, layers=layers, other=other[:4])))
