import json
pre = {}
for l in open('prelog.jsonl'):
    d = json.loads(l)
    pre[d['edit']] = dict(relmax=max(x['rel_max'] for x in d['layers']),
                          relsum=sum(x['rel_mean'] for x in d['layers']),
                          cap=any(c is not None and c > 1000 for x in d['layers'] for c in x['c']))
R = [json.loads(l) for l in open('behav_r45.jsonl')]
B = {(r['batch'], r['arm']): r for r in R}
agg = {}
for r in R:
    if r['arm'] == 'base' or not r['arm'].startswith('ac3_'): continue
    b = B[(r['batch'], 'base')]; a = agg.setdefault(r['arm'][4:], [0, 0, 0])
    a[0] += r['n']; a[1] += (r['mean'] - b['mean']) * r['n']; a[2] += (r['share8'] - b['share8']) * r['n']
beh = {k: (v[1] / v[0], v[2] / v[0]) for k, v in agg.items()}
# solve: mean of decision-rule pair (score_snapshot + b36_at_stop) and pooled per170 (r4_pool_1555)
pair = dict(u33=-1, u36=5, u63=3.5, u33_L5=1.5, u33_c6=0.5, late30=2, late60=3, pu=0, u33_ctrl=4,
            pe_lo=5, pe_hi=0.5, pe_L5=5, pe_c6=2, pe_ctrl=-2, pe_L14=-25, pe_L20=4.5, pe_L22=-1.5, pre15=1.5, pre35=0)
pool = dict(pe_lo=6.9, pe_L5=4.7, pe_c6=4.4, pe_L20=4.2, u33_ctrl=3.1, u63=2.8, u36=1.6, pre15=1.2, u33_c6=0.8,
            late60=0.5, late30=0.3, u33=0.0, pe_ctrl=0.0, u33_L5=-0.3, pe_hi=-0.3, pu=-0.5, pre35=-0.9, pe_L22=-2.7, pe_L14=-24.0)
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j+1]] == v[s[i]]: j += 1
        for k in range(i, j+1): r[s[k]] = (i + j) / 2
        i = j + 1
    return r
def sp(x, y):
    rx, ry = rank(x), rank(y); n = len(x); mx, my = sum(rx)/n, sum(ry)/n
    c = sum((a-mx)*(b-my) for a, b in zip(rx, ry)); return c / (sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry)) ** .5
def show(lbl, keys, fx, fy):
    x = [fx(k) for k in keys]; y = [fy(k) for k in keys]; print(f"{lbl:60} n={len(keys)} spearman={sp(x, y):+.2f}")
fam_pe = ['pe_lo', 'pe_hi', 'pe_L5', 'pe_c6', 'pe_L20', 'pe_L22', 'pre15', 'pre35']
noctrl = [k for k in pair if not k.endswith('ctrl') and k != 'pe_L14']
show('pe/pre family: relmax vs d mean calls-before-first-edit', fam_pe, lambda k: pre[k]['relmax'], lambda k: beh[k][0])
show('pe/pre family: relsum vs d mean calls-before-first-edit', fam_pe, lambda k: pre[k]['relsum'], lambda k: beh[k][0])
show('pe/pre family: relsum vs d share8', fam_pe, lambda k: pre[k]['relsum'], lambda k: beh[k][1])
show('all non-ctrl non-L14: relsum vs pooled solve per170', noctrl, lambda k: pre[k]['relsum'], lambda k: pool[k])
show('all non-ctrl non-L14: relsum vs pair-mean solve', noctrl, lambda k: pre[k]['relsum'], lambda k: pair[k])
show('all non-ctrl non-L14: |d mean cbfe| vs pooled solve', noctrl, lambda k: abs(beh[k][0]), lambda k: pool[k])
show('all non-ctrl non-L14: d share8 vs pooled solve', noctrl, lambda k: beh[k][1], lambda k: pool[k])
predmid = dict(u33=5.5, u36=5.5, u33_c6=5.5, pu=5.5, late30=2, late60=2, u63=1.5, u33_L5=5.5, pe_L5=2.5, pe_lo=2.5,
               pe_hi=0.5, pe_c6=3, pre15=1.5, pre35=2, pe_L14=1.5, pe_L20=3, pe_L22=1, u33_ctrl=0, pe_ctrl=0)
allk = list(pair)
show('predicted midpoint vs pair-mean solve (19 ac3 edits)', allk, lambda k: predmid[k], lambda k: pair[k])
show('predicted midpoint vs pooled per170 (19 ac3 edits)', allk, lambda k: predmid[k], lambda k: pool[k])
nl = [k for k in allk if k != 'pe_L14']
show('predicted midpoint vs pooled per170 (excl L14)', nl, lambda k: predmid[k], lambda k: pool[k])
for k in sorted(pre):
    if k in beh: print(f"{k:10} relmax {pre[k]['relmax']:.3f} relsum {pre[k]['relsum']:.3f} cap {pre[k]['cap']} dcbfe {beh[k][0]:+.2f} ds8 {beh[k][1]:+.3f} pair {pair.get(k)} pool {pool.get(k)}")
