import json,glob,os,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
pools={}
for f in glob.glob(R+'*/selection_rules/*.json'):
    d=json.load(open(f)); org=f.replace(chr(92),'/').split('/')[-3]
    b=os.path.basename(f)
    w=json.load(open(R+org+'/wide/'+b))
    ds='docvqa' if 'docvqa' in b else 'infovqa' if 'infovqa' in b else 'scifact'
    base=w['baseline']['ndcg@5']
    cand={c['label']:c['retention_labels_all'] for c in d['candidates']}
    res={}
    for r in d['results']:
        if r['reference']!='labels':continue
        res[(r['rule'],r['target'])]=r
    pools[(org,ds)]=dict(base=base,cand=cand,res=res,nq=d['n_queries'])
print(len(pools))
rules=sorted({k[0] for p in pools.values() for k in p['res']}); print(rules)
# noise
noise={}
for k,p in pools.items():
    resid={}
    for r in p['res'].values():
        for o in r['outcomes']:
            if o['selected'] in p['cand'] and o.get('holdout_labels') is not None:
                resid.setdefault(o['selected'],[]).append(o['holdout_labels']-p['cand'][o['selected']])
    # dedupe? splits repeat across rules for same seed; unique per (seed) would be better
    noise[k]=resid
for ds in ['docvqa','scifact','infovqa']:
    sds=[];bs=[]
    for k,p in pools.items():
        if k[1]!=ds:continue
        bs.append(p['base'])
        for c,v in noise[k].items():
            if len(v)>=4: sds.append(np.std(v,ddof=1))
    print(ds,'n cand',len(sds),'mean sd',np.mean(sds),'median sd',np.median(sds),'median base',np.median(bs),'npools',len(bs))
# pool-level
def pooled(k):
    s=[np.std(v,ddof=1) for v in noise[k].values() if len(v)>=4]
    return np.mean(s) if s else np.nan
ks=[k for k in pools if not np.isnan(pooled(k))]
x=np.array([1-pools[k]['base'] for k in ks]);y=np.array([pooled(k) for k in ks])
print('all',len(ks),stats.spearmanr(x,y))
for ds in ['docvqa','infovqa','scifact']:
    kk=[k for k in ks if k[1]==ds]
    print(ds,len(kk),stats.spearmanr([pools[k]['base'] for k in kk],[pooled(k) for k in kk]))
# miss rates paired
def miss(k,rule):
    m=[];
    for (ru,t),r in pools[k]['res'].items():
        if ru==rule: m.append(1-r['met'])
    return np.mean(m) if m else np.nan
models=sorted({k[0] for k in pools if k[1]=='docvqa'}&{k[0] for k in pools if k[1]=='infovqa'})
print('paired',len(models))
rng=np.random.default_rng(0)
for rule in rules:
    d=np.array([miss((m,'docvqa'),rule)-miss((m,'infovqa'),rule) for m in models])*100
    bs=[rng.choice(d,len(d)).mean() for _ in range(5000)]
    try: p=stats.wilcoxon(d).pvalue
    except Exception: p=np.nan
    print(rule,round(d.mean(),2),np.round(np.percentile(bs,[2.5,97.5]),2),p)
    for ds in ['docvqa','infovqa','scifact']:
        mm=[miss(k,rule) for k in pools if k[1]==ds]
        print('   ',ds,np.nanmean(mm)*100)
# drop-extreme / siblings sensitivity for 0.975
rule='lower 0.975'
d=np.array([miss((m,'docvqa'),rule)-miss((m,'infovqa'),rule) for m in models])*100
print(sorted(np.round(d,1)))
print('leave-one-out min/max',min(np.delete(d,i).mean() for i in range(len(d))),max(np.delete(d,i).mean() for i in range(len(d))))
fam=lambda m:m.split('--')[0]
for rule in ['lower 0.975','lower 0.99','point']:
    d={m:miss((m,'docvqa'),rule)-miss((m,'infovqa'),rule) for m in models}
    g={}
    for m,v in d.items(): g.setdefault(fam(m),[]).append(v)
    gv=np.array([np.mean(v) for v in g.values()])*100
    print(rule,'org-clustered',len(gv),gv.mean(),stats.wilcoxon(gv).pvalue, 'nonzero', (gv>0).sum())
# confound: baseline-controlled? within-model: baseline diff vs miss diff
bd=np.array([pools[(m,'docvqa')]['base']-pools[(m,'infovqa')]['base'] for m in models])
print('base diff range',bd.min(),bd.max())
