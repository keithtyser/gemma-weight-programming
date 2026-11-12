"""Layer scan of a prompt-rule contrast: how large and how consistent is the shift a rule writes at decisions?"""
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu"); h = a["h"].float(); meta = a["meta"]
rules = sorted({m["tag"] for m in meta})
idx = {(m["tag"], m["label"], m["pair"]): i for i, m in enumerate(meta)}
D = {}
for t in rules:
    pairs = sorted({m["pair"] for m in meta if m["tag"] == t and m["label"] == 1})
    tr = [p for p in pairs if meta[idx[(t, 1, p)]]["split"] == "train"]; ho = [p for p in pairs if p not in tr]
    d = lambda ps: h[[idx[(t, 1, p)] for p in ps]] - h[[idx[(t, 0, p)] for p in ps]]  # [n, L, D]
    dtr, dho = d(tr), d(ho); mu = dtr.mean(0); D[t] = mu
    print(t, "pairs train/heldout", len(tr), len(ho))
    print(" layer |mean diff| /rms(h)  heldout: cos(pair diff, train mean)  frac proj>0")
    for l in list(range(0, h.shape[1], 3)) + [h.shape[1] - 1]:
        u = mu[l] / mu[l].norm(); c = torch.nn.functional.cosine_similarity(dho[:, l], u[None], dim=-1)
        print(f"  {l:3d} {mu[l].norm():8.2f} {mu[l].norm() / h[:, l].norm(dim=-1).mean():6.3f}   {c.mean():.2f}   {(c > 0).float().mean():.2f}")
if len(rules) == 2:
    print("cos between rule directions by layer", [round(float(torch.nn.functional.cosine_similarity(D[rules[0]][l], D[rules[1]][l], dim=0)), 2) for l in range(0, h.shape[1], 3)])
