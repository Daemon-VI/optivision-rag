import json,glob,collections,numpy as np
B="C:/Users/Lenovo/projects/optivision-package/reports/universal/"
rng=np.random.default_rng(0)
def load(pat):
    cells=[]
    for f in sorted(glob.glob(B+pat)):
        d=json.load(open(f)); m=f.replace(chr(92),'/').split('/campaign/')[-1].split('/')[0] if 'campaign' in f else f
        for r in d['results']:
            cells.append(dict(model=m,file=f,ds=d['dataset'],ref=r['reference'],rule=r['rule'],t=r['target'],splits=r['splits'],met=r['met'],comp=r['compression_median'],hm=r['holdout_mean']))
    return cells
c=load('campaign/*/selection_rules/*.json')
print(len(c),len(set(x['file'] for x in c)),len(set(x['model'] for x in c)))
print(collections.Counter(x['ref'] for x in c)); print(collections.Counter(x['t'] for x in c))
lab=[x for x in c if x['ref']=='labels']
rules=sorted(set(x['rule'] for x in lab))
print({r:sum(x['splits'] for x in lab if x['rule']==r) for r in rules})
def miss(cs): 
    n=sum(x['splits'] for x in cs); return 1-sum(x['met']*x['splits'] for x in cs)/n
for r in rules: print(r,round(miss([x for x in lab if x['rule']==r]),4), {t:round(1-np.mean([x['met'] for x in lab if x['rule']==r and x['t']==t]),3) for t in sorted(set(x['t'] for x in lab))})
# cluster bootstrap by model
models=sorted(set(x['model'] for x in lab))
def bs(fn,B_=2000):
    v=[]
    for _ in range(B_):
        ms=rng.choice(models,len(models)); cs=[]
        for m in ms: cs+= [x for x in lab if x['model']==m]
        v.append(fn(cs))
    return np.percentile(v,[2.5,97.5])
for r in ['lower 0.975','lower 0.99','lower 0.999','point','calibrate() default']:
    print(r,'miss CI',bs(lambda cs:miss([x for x in cs if x['rule']==r]),500))
# paired with default
key=lambda x:(x['file'],x['t'])
D={key(x):x for x in lab if x['rule']=='calibrate() default'}
by=collections.defaultdict(list)
for x in lab: by[x['rule']].append(x)
def paired(r,cs):
    cr=[];dm=[]
    for x in cs:
        if x['rule']==r and key(x) in D and x['comp'] and D[key(x)]['comp']:
            d=[y for y in cs if y['rule']=='calibrate() default' and key(y)==key(x)][0]
            cr.append(x['comp']/d['comp']); dm.append(x['met']-d['met'])
    return np.exp(np.mean(np.log(cr))),-100*np.mean(dm)
for r in ['point','lower 0.975','lower 0.99']:
    est=paired(r,lab)
    v=[]
    for _ in range(300):
        cs=[]
        for m in rng.choice(models,len(models)): cs+=[x for x in lab if x['model']==m]
        v.append(paired(r,cs))
    v=np.array(v); print(r,est,np.percentile(v,[2.5,97.5],axis=0).T.tolist())
# subsets: by dataset, drop duplicates
for ds in sorted(set(x['ds'] for x in lab)):
    print(ds,{r:round(miss([x for x in lab if x['ds']==ds and x['rule']==r]),3) for r in ['calibrate() default','lower 0.975','lower 0.99','lower 0.999','point']})
# per-model miss lower .975 spread; leave extreme out
pm={m:miss([x for x in lab if x['model']==m and x['rule']=='lower 0.975']) for m in models}
print(sorted(pm.items(),key=lambda a:-a[1])[:8], np.median(list(pm.values())))
print('share models with zero miss lower .975',np.mean([v==0 for v in pm.values()]))
# drop top 5 models
top=set(m for m,_ in sorted(pm.items(),key=lambda a:-a[1])[:5])
print('drop top5',{r:round(miss([x for x in lab if x['model'] not in top and x['rule']==r]),3) for r in ['calibrate() default','lower 0.975','lower 0.99','lower 0.999','point']})
# point at targets
print({t:round(np.mean([x['met'] for x in lab if x['rule']=='point' and x['t']==t]),3) for t in sorted(set(x['t'] for x in lab))})
# nonlabel refs
for ref in sorted(set(x['ref'] for x in c)): print(ref,{r:round(miss([x for x in c if x['ref']==ref and x['rule']==r]),3) for r in ['calibrate() default','lower 0.975','point']})
