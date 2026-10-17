import json,glob,os,collections
import numpy as np
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
files=sorted(glob.glob(R+"*/selection_rules/*.json"))
rows=[]
for f in files:
    d=json.load(open(f)); model=f.replace("\\","/").split("/campaign/")[1].split("/")[0]
    ds='docvqa' if 'docvqa' in d['dataset'] else 'infovqa' if 'infovqa' in d['dataset'] else 'scifact'
    for r in d['results']:
        rows.append(dict(model=model,ds=ds,ref=r['reference'],rule=r['rule'],t=r['target'],met=r['met'],sel=r['selected'],comp=r['compression_median'],hm=r['holdout_mean'],splits=r['splits']))
print(len(files),len(rows),collections.Counter(r['ref'] for r in rows),collections.Counter(r['t'] for r in rows))
print(collections.Counter(r['ds'] for r in rows)); print(len({r['model'] for r in rows}))
rng=np.random.default_rng(0)
models=sorted({r['model'] for r in rows})
def agg(sub,rule,t,ref='labels'):
    x=[r for r in sub if r['rule']==rule and r['t']==t and r['ref']==ref]
    return x
rules=sorted({r['rule'] for r in rows})
for ref in ['labels']:
  for rule in rules:
    out=[]
    for t in [0.95,0.97,0.99]:
        x=agg(rows,rule,t,ref)
        if not x: out.append('NA');continue
        # model-cluster bootstrap
        bym=collections.defaultdict(list)
        for r in x: bym[r['model']].append(r['met'])
        ms=[np.mean(v) for v in bym.values()]
        bs=[np.mean(rng.choice(ms,len(ms))) for _ in range(3000)]
        out.append(f"{np.mean([r['met'] for r in x]):.3f} (n={len(x)}, m={len(ms)}) [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}] ge975={np.mean([r['met']>=0.975 for r in x]):.2f}")
    print(rule,out)
# by dataset
for ds in ['docvqa','infovqa','scifact']:
    sub=[r for r in rows if r['ds']==ds]
    print(ds,{rule:[round(np.mean([r['met'] for r in agg(sub,rule,t)]),3) for t in [0.95,0.97,0.99]] for rule in ['point','lower 0.975','calibrate() default']})
# ratio
d={ (r['model'],r['ds'],r['t'],r['rule']):r for r in rows if r['ref']=='labels'}
for t in [0.95,0.97,0.99]:
    rat=[d[(m,ds,tt,'point')]['comp']/d[(m,ds,tt,'calibrate() default')]['comp'] for (m,ds,tt,ru) in d if ru=='point' and tt==t]
    print(t,np.median(rat),len(rat))
# default failures
for r in rows:
    if r['rule']=='calibrate() default' and r['ref']=='labels' and r['met']<0.9: print('fail',r['model'],r['ds'],r['t'],r['met'])
# point: holdout_mean vs target, "no selection" cases
print(collections.Counter((r['sel'] is None) for r in rows))
# drop siblings: one per family-ish prefix; leave-one-out extremes
x=agg(rows,'point',0.99); print(sorted([(round(r['met'],2),r['model'][:30],r['ds']) for r in x])[:3],sorted([(round(r['met'],2),r['model'][:30]) for r in x])[-5:])
# leave-one-model-out for lower 0.975
for rule in ['point','lower 0.975']:
  for t in [0.95,0.99]:
    x=agg(rows,rule,t); 
    lo=[np.mean([r['met'] for r in x if r['model']!=m]) for m in models]
    print(rule,t,min(lo),max(lo))
# how is 'met' defined: check against outcomes
f=files[0]; d0=json.load(open(f)); r=[r for r in d0['results'] if r['rule']=='point' and r['target']==0.99 and r['reference']=='labels'][0]
print(r['met'],r['met_labels'],[o['holdout'] for o in r['outcomes']][:5],[o['calibration'] for o in r['outcomes']][:5])
print(np.mean([o['holdout']>=0.99 for o in r['outcomes']]))
