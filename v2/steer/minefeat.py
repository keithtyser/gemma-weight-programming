"""Within-task link between behaviours and solving (CPU). Reads <mine>/rollouts.jsonl + calls.jsonl.gz."""
import sys, json, gzip, collections, math
mine, srcs = sys.argv[1], set(sys.argv[2].split(',')) if len(sys.argv) > 2 and sys.argv[2] != 'all' else None
ros = {}
for l in open(mine + '/rollouts.jsonl'):
    r = json.loads(l)
    if r['infra'] or r.get('resolved') is None or (srcs and r['src'] not in srcs): continue
    ros[(r['src'], r['run_key'])] = r
calls = collections.defaultdict(list)
for l in gzip.open(mine + '/calls.jsonl.gz', 'rt'):
    c = json.loads(l)
    k = (c['src'], c['run_key'])
    if k in ros and c['is_agent'] is not False: calls[k].append(c)
print('srcs', collections.Counter(k[0] for k in ros).most_common(12))
E = ('edit_file', 'write_file')
def feats(cs, r):
    cs.sort(key=lambda c: c['call'])
    T = [tuple(c.get('tools') or ()) for c in cs]
    fe = next((i for i, t in enumerate(T) if any(x in E for x in t)), None)
    f = {}
    f['edited'] = fe is not None
    if fe is None: return f
    pre, post = T[:fe], T[fe + 1:]
    le = max(i for i, t in enumerate(T) if any(x in E for x in t))
    f['calls_before_edit>=3'] = fe >= 3
    f['calls_before_edit>=5'] = fe >= 5
    f['calls_before_edit>=8'] = fe >= 8
    f['reads_before_edit>=2'] = sum('read_file' in t for t in pre) >= 2
    f['runs_before_edit>=2'] = sum('run_command' in t for t in pre) >= 2
    f['runs_before_edit>=4'] = sum('run_command' in t for t in pre) >= 4
    f['run_right_before_edit'] = fe > 0 and 'run_command' in T[fe - 1]
    f['read_right_before_edit'] = fe > 0 and 'read_file' in T[fe - 1]
    f['run_after_last_edit'] = any('run_command' in t for t in T[le + 1:])
    f['runs_after_last_edit>=2'] = sum('run_command' in t for t in T[le + 1:]) >= 2
    f['read_after_last_edit'] = any('read_file' in t for t in T[le + 1:])
    ne = sum(any(x in E for x in t) for t in T)
    f['edits>=2'] = ne >= 2
    f['edits>=3'] = ne >= 3
    f['edits>=5'] = ne >= 5
    f['used_write_file'] = any('write_file' in t for t in T)
    f['multi_tool_call'] = any(len(t) > 1 for t in T)
    f['submit_called'] = bool(r.get('submit_called'))
    f['total_calls>=10'] = len(T) >= 10
    f['total_calls>=15'] = len(T) >= 15
    f['first_think>=300'] = (cs[0].get('n_think') or 0) >= 300
    f['mean_think>=300'] = sum(c.get('n_think') or 0 for c in cs) / len(cs) >= 300
    f['edit_think>=400'] = (cs[fe].get('n_think') or 0) >= 400
    f['any_forced'] = any(c.get('forced') for c in cs)
    f['any_loop'] = any(c.get('loop_onset') is not None for c in cs)
    return f
by = collections.defaultdict(list)
for k, cs in calls.items():
    r = ros[k]; by[r['instance_id']].append((bool(r['resolved']), feats(cs, r)))
mixed = {t: v for t, v in by.items() if any(a for a, _ in v) and any(not a for a, _ in v)}
print('rollouts', sum(len(v) for v in by.values()), 'solved', sum(a for v in by.values() for a, _ in v), 'tasks', len(by), 'mixed tasks', len(mixed))
names = sorted({n for v in by.values() for _, f in v for n in f})
out = []
for n in names:
    # Mantel-Haenszel within task: odds ratio of solving given feature
    num = den = 0.0; var = 0.0; dsum = 0.0; base = tot = 0
    for t, v in mixed.items():
        v = [(a, f[n]) for a, f in v if n in f]
        N = len(v)
        if N < 2: continue
        a = sum(1 for s, x in v if s and x); b = sum(1 for s, x in v if not s and x)
        c = sum(1 for s, x in v if s and not x); d = sum(1 for s, x in v if not s and not x)
        num += a * d / N; den += b * c / N
        n1, m1 = a + b, a + c
        if N > 1 and 0 < n1 < N:
            dsum += a - n1 * m1 / N; var += n1 * (N - n1) * m1 * (N - m1) / (N * N * (N - 1))
        base += a + b; tot += N
    if den == 0 or var == 0: continue
    z = dsum / math.sqrt(var)
    out.append((z, n, num / den, base / tot))
for z, n, orr, fr in sorted(out, reverse=True):
    print(f'{n:28s} OR {orr:5.2f}  z {z:+6.1f}  freq {fr:.2f}')
