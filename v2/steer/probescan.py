"""Heldout AUC of a trained L2 logistic probe per layer and tag (stronger read than the class-mean difference).
Usage: probescan.py ACTS.pt tag[,tag...] [step]   Prints li:trainAUC>heldoutAUC for weight decay lam in (10, 0.1, 0.001)."""
import sys, torch
acts = torch.load(sys.argv[1], map_location="cpu"); meta = acts["meta"]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 3
def auc(s, y): return float((s[y == 1][:, None] > s[y == 0][None, :]).float().mean())
def fit(X, y, lam):
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + lam * (w * w).sum() / 2
        loss.backward(); return loss
    opt.step(closure); return w.detach(), b.detach()
for tag in sys.argv[2].split(","):
    idx = lambda sp: [i for i, m in enumerate(meta) if m["tag"] == tag and m["split"] == sp]
    tr, ho = idx("train"), idx("heldout")
    ytr, yho = (torch.tensor([float(meta[i]["label"]) for i in ix]) for ix in (tr, ho))
    for lam in (10.0, 0.1, 0.001):
        out = []
        for li in range(0, acts["h"].shape[1], step):
            Z = acts["h"][:, li].float()
            mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-6
            Xtr, Xho = (Z[tr] - mu) / sd, (Z[ho] - mu) / sd
            w, b = fit(Xtr, ytr, lam)
            out.append(f"{li}:{auc(Xtr @ w + b, ytr):.2f}>{auc(Xho @ w + b, yho):.2f}")
        print(tag, f"lam={lam}", len(tr), len(ho), " ".join(out), flush=True)
