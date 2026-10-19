import json,glob,numpy as np,os
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
rng=np.random.default_rng(1)
rows=[]  # model,ds,rule,target,seed,met,comp
nomet=0;chk=0
for f in sorted(glob.glob(R+'*/selection_rules/*.json')):
    m=os.path.basename(os.path.dirname(os.path.dirname(f)));d=json.load(open(f))
    for r in d['results']:
        if r['reference']!='labels':continue
        for o in r['outcomes']:
            h=o.get('holdout')
            rows.append((m,d['dataset'],r['rule'],r['target'],o['seed'],None if h is None else float(h>=r['target']),o.get('compression',2.0),h))
        mm=np.mean([ (o['holdout']>=r['target']) for o in r['outcomes'] if o.get('holdout') is not None] or [0])
        chk+=abs(mm-r['met'])>1e-9
print('rows',len(rows),'none',sum(x[5] is None for x in rows),'met mismatch cells',chk)
import collections
rules=sorted(set(x[2] for x in rows)); ms=sorted(set(x[0] for x in rows))
def agg(rule,idx_models,target=None):
    a=[];c=[]
    for x in rows:
        if x[2]==rule and (target is None or x[3]==target):
            a.append(x);
    return a
byrm=collections.defaultdict(list)
for x in rows: byrm[(x[2],x[0])].append(x)
for rule in rules:
    pm=np.array([np.mean([y[5] for y in byrm[(rule,m)] if y[5] is not None]) for m in ms])
    def st(idx):
        v=[y for i in idx for y in byrm[(rule,ms[i])]]
        return np.mean([y[5] for y in v if y[5] is not None]), np.exp(np.mean([np.log(y[6]) for y in v]))
    p=st(range(len(ms)));bs=np.array([st(rng.integers(0,len(ms),len(ms))) for _ in range(1000)])
    lo,hi=np.percentile(bs[:,0],[2.5,97.5])
    pt={t:round(np.mean([y[5] for m in ms for y in byrm[(rule,m)] if y[3]==t and y[5] is not None]),3) for t in (.95,.97,.99)}
    pg={t:round(np.exp(np.mean([np.log(y[6]) for m in ms for y in byrm[(rule,m)] if y[3]==t])),1) for t in (.95,.97,.99)}
    print(f'{rule:28s} met {p[0]:.3f} [{lo:.3f},{hi:.3f}] gm {p[1]:.1f} below {int((pm<.95).sum())} min {pm.min():.3f} {ms[pm.argmin()][:25]} {pt} {pg}')
# paired cell
cell=collections.defaultdict(dict)
for x in rows: cell[(x[0],x[1],x[3])].setdefault(x[2],[]).append(x)
def cellstats(rule):
    return {k:(np.mean([np.log(y[6]) for y in v[rule]]),np.mean([1-y[5] for y in v[rule] if y[5] is not None])) for k,v in cell.items()}
cs={r:cellstats(r) for r in rules}
keys=sorted(cell); print(len(keys))
mi={m:i for i,m in enumerate(ms)}
def paired(a,b,sub=None):
    ks=[k for k in keys if sub is None or sub(k)]
    dl=np.array([cs[a][k][0]-cs[b][k][0] for k in ks]);dm=np.array([cs[a][k][1]-cs[b][k][1] for k in ks])
    mod=np.array([mi[k[0]] for k in ks])
    def f(idx):
        # cluster by model
        sel=np.concatenate([np.where(mod==i)[0] for i in idx]);return np.exp(dl[sel].mean()),dm[sel].mean()*100
    p=f(range(len(ms)));bs=np.array([f(rng.integers(0,len(ms),len(ms))) for _ in range(1000)])
    print(a,'/',b,len(ks),'ratio %.3f [%.3f,%.3f]'%(p[0],*np.percentile(bs[:,0],[2.5,97.5])),'miss diff %.2f [%.2f,%.2f]'%(p[1],*np.percentile(bs[:,1],[2.5,97.5])))
paired('lower 0.999','calibrate() default');paired('lower 0.999','bonferroni 0.975');paired('bonferroni 0.975','calibrate() default')
# leave-out sensitivity
ex=lambda s:(lambda k:s not in k[0])
for s in ['colpali-v1.2','jina']: 
    print('excluding',s);paired('lower 0.999','calibrate() default',ex(s))
# by dataset type
paired('lower 0.999','calibrate() default',lambda k:'scifact' in k[1].lower())
paired('lower 0.999','calibrate() default',lambda k:'scifact' not in k[1].lower())
# is selection-feasible fallback? count of cells where rule selects nothing
print(set(x[1] for x in rows))
print('abstention share (no selection) by rule, per target')
for rule in rules:
    print(rule,{t:round(np.mean([y[7] is None for y in rows if y[2]==rule and y[3]==t]),3) for t in (.95,.97,.99)})
# abstention-as-fail sensitivity: met counting abstain as miss
for rule in rules:
    pm=np.array([np.mean([ (y[5] if y[5] is not None else 0.0) for y in byrm[(rule,m)]]) for m in ms])
    print('abstain=miss',rule,'pooled %.3f'%np.mean([ (y[5] if y[5] is not None else 0.0) for y in rows if y[2]==rule]),'models<.95',int((pm<.95).sum()))
print('--- per-target / subset cluster CI of met')
def ci(rule,t=None,sub=lambda m,ds:True):
    mset=[m for m in ms if any(sub(m,y[1]) for y in byrm[(rule,m)])]
    d={m:[y[5] for y in byrm[(rule,m)] if y[5] is not None and (t is None or y[3]==t) and sub(m,y[1])] for m in mset}
    def f(idx):
        v=np.concatenate([d[mset[i]] for i in idx]);return v.mean()
    p=f(range(len(mset)));b=[f(rng.integers(0,len(mset),len(mset))) for _ in range(1000)]
    return round(p,3),[round(x,3) for x in np.percentile(b,[2.5,97.5])]
for rule in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.975 + margin 0.01']:
    print(rule,[ (t,)+ci(rule,t) for t in (.95,.97,.99)])
    print('  text',ci(rule,None,lambda m,ds:'scifact' in ds),'image',ci(rule,None,lambda m,ds:'scifact' not in ds))
    print('  no colpali/colqwen siblings-ish',ci(rule,None,lambda m,ds:'colpali' not in m))
