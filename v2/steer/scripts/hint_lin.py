"""How much of a prompt rule's residual shift is a constant, and how much a low-rank linear map of the MLP input?"""
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu"); h = a["h"].float(); x = a["xin"].float(); meta = a["meta"]
idx = {(m["tag"], m["label"], m["pair"]): i for i, m in enumerate(meta)}
for t in sorted({m["tag"] for m in meta}):
    pairs = sorted({m["pair"] for m in meta if m["tag"] == t and m["label"] == 1})
    tr = [p for p in pairs if meta[idx[(t, 1, p)]]["split"] == "train"]; ho = [p for p in pairs if p not in tr]
    i1 = lambda ps: [idx[(t, 1, p)] for p in ps]; i0 = lambda ps: [idx[(t, 0, p)] for p in ps]
    for l in (12, 15, 18, 21, 24):
        Ttr, Tho = h[i1(tr), l] - h[i0(tr), l], h[i1(ho), l] - h[i0(ho), l]
        Xtr, Xho = x[i0(tr), l], x[i0(ho), l]
        tot = (Tho ** 2).sum()
        r2 = lambda P: float(1 - ((Tho - P) ** 2).sum() / tot)
        out = [f"const {r2(Ttr.mean(0, keepdim=True).expand_as(Tho)):.2f}"]
        K = Xtr @ Xtr.T
        for lam in (0.01, 0.1, 1.0):
            Wd = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(tr)), Ttr)  # dual coefficients [n, D]
            full = Xho @ Xtr.T @ Wd
            s = [f"full {r2(full):.2f}"]
            W = Xtr.T @ Wd  # [D_in, D_out] map
            U, S, V = torch.svd_lowrank(W, q=32)
            for k in (1, 2, 4, 8, 16):
                s.append(f"k{k} {r2(Xho @ U[:, :k] * S[:k] @ V[:, :k].T):.2f}")
            out.append(f"ridge {lam}: " + " ".join(s))
        print(t, "layer", l, "| heldout R2 |", " | ".join(out))
