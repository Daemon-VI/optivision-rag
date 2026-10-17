import os,json,glob,numpy as np,collections,re
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
fs=glob.glob(R+'*/wide/*.json')
data=collections.defaultdict(dict)
for f in fs:
    j=json.load(open(f)); mod=os.path.normpath(f).replace(os.sep,'/').split('/campaign/')[1].split('/')[0]
    ds='doc' if 'docvqa' in j['dataset'].lower() else 'info' if 'infovqa' in j['dataset'].lower() else None
    if not ds: continue
    rows={}
    for r in j['rows']:
        s=r['pipeline']['stages'] if isinstance(r['pipeline'],dict) else r['pipeline']
        if len(s)==1 and s[0].get('stage')=='project' and s[0].get('method','pca')=='pca':
            rows[s[0]['dim']]=(r['retention:ndcg@5'],r.get('retention_ci:ndcg@5'))
    data[mod][ds]=dict(rows=rows,base=j['baseline']['ndcg@5'],dim=j['baseline']['dim'],vpd=j['baseline']['vectors_per_doc'],nq=j['n_queries'])
pairs=[]
for mod,d in data.items():
    if 'doc' in d and 'info' in d:
        for k in set(d['doc']['rows'])&set(d['info']['rows']):
            pairs.append((mod,k,d['doc']['rows'][k][0],d['info']['rows'][k][0],d['doc']['dim']))
print(len(data),len(pairs),len({p[0] for p in pairs}))
print(sorted(collections.Counter(p[1] for p in pairs).items()))
rng=np.random.default_rng(1)
def summ(ps,name):
    mods=sorted({p[0] for p in ps})
    if not mods: print(name,'none');return
    per=np.array([np.mean([p[3]-p[2] for p in ps if p[0]==mm]) for mm in mods])
    diffs=np.array([p[3]-p[2] for p in ps])
    bs=[np.mean(rng.choice(per,len(per))) for _ in range(5000)]
    w=stats.wilcoxon(diffs).pvalue if len(diffs)>5 else np.nan
    w2=stats.wilcoxon(per).pvalue if len(per)>5 else np.nan
    print(f"{name}: pairs={len(ps)} models={len(mods)} pair-mean={diffs.mean()*100:.2f} model-mean={per.mean()*100:.2f} CI[{np.percentile(bs,2.5)*100:.2f},{np.percentile(bs,97.5)*100:.2f}] median={np.median(per)*100:.2f} frac>0 pairs={np.mean(diffs>0):.2f} models={np.mean(per>0):.2f} P_pairs={w:.2g} P_models={w2:.2g}")
    return per
for lo,hi in [(0,40),(40,70),(70,100),(100,200),(200,10**6)]: summ([p for p in pairs if lo<p[1]<=hi],f"k({lo},{hi}]")
for k in sorted({p[1] for p in pairs}): summ([p for p in pairs if p[1]==k],f"k={k}")
print('--- 128-d only')
for k in (8,16,24,32,48,64):summ([p for p in pairs if p[1]==k and p[4]==128],f"128d k={k}")
print('mean ret 128d k16/32', [ (k,np.mean([p[2] for p in pairs if p[1]==k and p[4]==128]),np.mean([p[3] for p in pairs if p[1]==k and p[4]==128])) for k in (16,32)])
# by family
fam=lambda mod: re.split('--',mod)[0]
ps=[p for p in pairs if p[1]<=40]
print('per-model small-k gap:')
pm=collections.defaultdict(list)
for p in ps: pm[p[0]].append(p[3]-p[2])
for k,v in sorted(pm.items(),key=lambda x:np.mean(x[1])): print(f"  {k:55s} {np.mean(v)*100:6.1f} n={len(v)} base d={data[k]['doc']['base']:.3f} i={data[k]['info']['base']:.3f}")
vals=np.array([np.mean(v) for v in pm.values()])
print('model-level',vals.mean()*100,np.median(vals)*100,'trim-top2/bottom2',np.mean(np.sort(vals)[2:-2])*100)
# family-level
ff=collections.defaultdict(list)
for k,v in pm.items(): ff[fam(k)].append(np.mean(v))
fv=np.array([np.mean(v) for v in ff.values()]);print('families',len(fv),fv.mean()*100,(fv>0).mean(),stats.wilcoxon(fv).pvalue)
# retention-in-loss ratio: (1-ret_info)/(1-ret_doc)
r=[(1-p[3])/(1-p[2]) for p in ps if p[2]<0.97]
print('loss ratio info/doc median',np.median(r),len(r))
# gap vs baseline docvqa and info
g=[np.mean(pm[m]) for m in pm]; bd=[data[m]['doc']['base'] for m in pm]; bi=[data[m]['info']['base'] for m in pm]
print('spearman gap~docbase',stats.spearmanr(g,bd),'gap~infobase',stats.spearmanr(g,bi))
# paired-by-model CI-aware: query-level overlap - fraction with nonoverlapping CIs
cnt=0;tot=0
for mod,d in data.items():
    if 'doc' in d and 'info' in d:
        for k in set(d['doc']['rows'])&set(d['info']['rows']):
            if k<=40:
                tot+=1;a=d['doc']['rows'][k][1];b=d['info']['rows'][k][1]
                if a and b and b[0]>a[1]: cnt+=1
print('nonoverlap',cnt,tot)
print('nq',{m:(d['doc']['nq'],d['info']['nq']) for m,d in list(data.items())[:2]})
