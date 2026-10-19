import json,numpy as np,collections
from scipy.stats import beta
from optivision.adapters import MODELS
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
info={m.model_id:m for m in MODELS.values()} if isinstance(MODELS,dict) else {m.model_id:m for m in MODELS}
units=[]
for mid,v in d.items():
    mi=info[mid]
    for ds,x in v.items():
        rows={r['label']:r for r in x['rows'] if r['label']!='float32'}
        units.append(dict(model=mid,ds=ds,dim=mi.dimension,mod=mi.modality,rows=rows,fp16='float16' in (x.get('note') or '')))
print(len(units),collections.Counter((u['dim'],u['mod']) for u in units))
rng=np.random.default_rng(0)
def run(t,c=0.9,q=0.10,excl=None,prior=(1,1),pt=False):
    res=[]
    for u in units:
        if excl and u['model'] in excl: continue
        peers=[p for p in units if p['model']!=u['model'] and p['dim']==u['dim'] and p['mod']==u['mod'] and not(excl and p['model'] in excl)]
        if not peers: continue
        best=None
        for lab,r in u['rows'].items():
            ps=[p['rows'][lab] for p in peers if lab in p['rows']]
            n=len(ps)
            if n==0: continue
            k=sum(x['retention']>=t for x in ps)
            lo=beta.ppf(q,prior[0]+k,prior[1]+n-k)
            if lo>=c:
                comp=np.median([x['compression'] for x in ps])
                if best is None or comp>best[0]: best=(comp,lab)
        if best is None: res.append((u['model'],None,1,1,1,1));continue
        r=u['rows'][best[1]]
        ob=max(x['compression'] for x in u['rows'].values() if x['retention']>=t) if any(x['retention']>=t for x in u['rows'].values()) else 1
        res.append((u['model'],best[1],r['retention']>=t,r['retention_interval'][0]>=t,r['compression'],ob,r['retention']))
    return res
def summ(res,t):
    sel=[r for r in res if r[1] is not None]
    models=sorted(set(r[0] for r in sel))
    hit=np.array([r[2] for r in sel],float);lo=np.array([r[3] for r in sel],float)
    byM=collections.defaultdict(list)
    for i,r in enumerate(sel): byM[r[0]].append(i)
    bs=[]
    for _ in range(2000):
        ms=rng.choice(models,len(models));idx=[i for m in ms for i in byM[m]]
        bs.append(hit[idx].mean())
    fails=[t-r[6] for r in sel if not r[2]]
    return dict(n=len(sel),nres=len(res),hit=hit.mean(),ci=np.percentile(bs,[2.5,97.5]).round(3),lo=lo.mean(),comp=np.median([r[4] for r in sel]),own=np.median([r[5] for r in sel]),fails=len(fails),medshort=np.median(fails) if fails else 0,max=max(fails) if fails else 0)
for t in (.95,.97,.99):
    print(t,summ(run(t),t))
print('c=.95')
for t in (.95,.97,.99): print(t,summ(run(t,c=.95),t))
print('no fp16 models')
fp=set(u['model'] for u in units if u['fp16']);print(len(fp))
for t in (.95,.97,.99): print(t,summ(run(t,excl=fp),t))
print('jeffreys prior')
for t in (.95,.97,.99): print(t,summ(run(t,prior=(.5,.5)),t))
