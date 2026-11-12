"""Batch-level test: per batch, mean solved over explore-more arms (0.125-0.5) minus the unedited arm of that batch,
and minus the mean of that batch's matched controls. Batches are the independent unit. Usage: batch_level.py results/runs.tsv"""
import json, collections, math, sys, statistics as st
R = collections.defaultdict(dict)
for l in open(sys.argv[1], encoding='utf-8'):
    a, j = l.split('\t', 1); d = json.loads(j); k = d.get('instance_id')
    if k: R[a][k] = bool(d.get('resolved'))
S = {a: sum(v.values()) for a, v in R.items()}
E = ('bneg_t0.125', 'bneg_t0.25', 'bneg_t0.5')
ctl = lambda a: a.startswith('bnm25') or a == 'bnegmatch_t0.5'
def t(xs):
    n = len(xs); m = st.mean(xs); s = st.stdev(xs); tv = m / (s / math.sqrt(n))
    return n, round(m, 2), round(s, 2), round(tv, 2)
def sign(xs):
    w = sum(x > 0 for x in xs); l = sum(x < 0 for x in xs); n = w + l
    return w, l, round(min(1, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n), 4)
db, dc = [], []
for i in range(3, 16):
    b = 'b%d' % i
    e = [S[b + '_' + x] for x in E if b + '_' + x in S]
    c = [S[a] for a in S if a.startswith(b + '_') and ctl(a[len(b) + 1:])]
    if not e: continue
    db.append(st.mean(e) - S[b + '_base'])
    if c: dc.append(st.mean(e) - st.mean(c))
    print(b, 'edit', e, 'base', S[b + '_base'], 'ctl', c)
print('edit - base per batch', [round(x, 1) for x in db], t(db), sign(db))
print('edit - ctl per batch', [round(x, 1) for x in dc], t(dc), sign(dc))
