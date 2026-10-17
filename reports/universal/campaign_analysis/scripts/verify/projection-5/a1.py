import json,glob,numpy as np,collections
from scipy import stats
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
def proj(v):
    out={}
    for r in v['rows']:
        s=r['pipeline']['stages']
        if len(s)==1 and s[0]['stage']=='project' and s[0].get('method')=='pca':
            out[s[0]['dim']]=r['retention']
    return out
pairs=[]
for mod,d in m.items():
    if D in d and I in d:
        a,b=proj(d[D]),proj(d[I])
        for k in sorted(set(a)&set(b)): pairs.append((mod,k,a[k],b[k]))
print(len(pairs),len({p[0] for p in pairs}))
print(collections.Counter(p[1] for p in pairs))
rng=np.random.default_rng(0)
def fam(mod): return mod.split('/')[0]
def summ(ps,name):
    if not ps: return
    mods=sorted({p[0] for p in ps}); 
    per={mm:np.mean([p[3]-p[2] for p in ps if p[0]==mm]) for mm in mods}
    diffs=np.array([p[3]-p[2] for p in ps]);
    x=np.array(list(per.values()))
    bs=[np.mean(rng.choice(x,len(x))) for _ in range(5000)]
    try: w=stats.wilcoxon(diffs).pvalue
    except: w=np.nan
    try: w2=stats.wilcoxon(x).pvalue
    except: w2=np.nan
    print(f"{name}: pairs={len(ps)} models={len(mods)} meanpair={diffs.mean()*100:.2f} model-mean={x.mean()*100:.2f} CI[{np.percentile(bs,2.5)*100:.2f},{np.percentile(bs,97.5)*100:.2f}] frac>0 pairs={np.mean(diffs>0):.2f} models={np.mean(x>0):.2f} wilcP pairs={w:.2g} models={w2:.2g}")
bk=[(0,40),(40,70),(70,100),(100,200),(200,10**6)]
for lo,hi in bk: summ([p for p in pairs if lo<p[1]<=hi],f"k({lo},{hi}]")
for k in sorted({p[1] for p in pairs}): summ([p for p in pairs if p[1]==k],f"k={k}")
# per model k<=40
print("--- ex-extreme / siblings")
ps=[p for p in pairs if p[1]<=40]
byf=collections.defaultdict(list)
for p in ps: byf[fam(p[0])].append(p)
print({f:round(np.mean([q[3]-q[2] for q in v])*100,1) for f,v in byf.items()})
# leave-one-family-out and family-level mean
fm=np.array([np.mean([q[3]-q[2] for q in v]) for v in byf.values()]);print('fam-mean',fm.mean()*100,len(fm),stats.wilcoxon(fm).pvalue)
# baselines
for mod in list(m)[:0]:pass
# ceiling: relation to retention level
for k in (16,32): 
    pk=[p for p in pairs if p[1]==k]
    print(k,'docvqa mean',np.mean([p[2] for p in pk]),'info',np.mean([p[3] for p in pk]))
# gap vs baseline docvqa
bd={mod:(m[mod][D]['baseline'],m[mod][I]['baseline'],'float16' in m[mod][D].get('note','')) for mod in {p[0] for p in pairs}}
ps=[p for p in pairs if p[1]<=40]
g=[p[3]-p[2] for p in ps]; b=[bd[p[0]][0] for p in ps]
print('spearman gap vs docvqa baseline',stats.spearmanr(g,b))
f16=[p for p in ps if bd[p[0]][2]]; n16=[p for p in ps if not bd[p[0]][2]]
summ(f16,'f16');summ(n16,'f32')
# mean baselines
print('mean baselines',np.mean([v[0] for v in bd.values()]),np.mean([v[1] for v in bd.values()]))
# retention in absolute-loss terms: loss relative scale
for p in ps[:0]:pass
