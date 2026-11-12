"""Heldout AUC of the class-mean-difference direction per layer and tag. Usage: aucscan.py ACTS.pt tag[,tag...]"""
import sys, torch
acts = torch.load(sys.argv[1], map_location="cpu"); meta = acts["meta"]
def auc(p, n): return float((p[:, None] > n[None, :]).float().mean())
for tag in sys.argv[2].split(","):
    sel = lambda sp, lab: torch.tensor([i for i, m in enumerate(meta) if m["tag"] == tag and m["split"] == sp and m["label"] == lab])
    t1, t0, h1, h0 = sel("train", 1), sel("train", 0), sel("heldout", 1), sel("heldout", 0)
    out = []
    for li in range(acts["h"].shape[1]):
        r = []
        for key in ("h", "xin"):
            Z = acts[key][:, li].float(); d = Z[t1].mean(0) - Z[t0].mean(0)
            r.append(auc(Z[h1] @ d, Z[h0] @ d))
        out.append(f"{li}:{r[0]:.2f}/{r[1]:.2f}")
    print(tag, len(t1), len(t0), len(h1), len(h0), " ".join(out), flush=True)
