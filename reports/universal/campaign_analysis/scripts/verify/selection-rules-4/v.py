import json,glob,numpy as np,pandas as pd
rows=[]
for f in glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/selection_rules/*.json'):
    d=json.load(open(f)); model=f.replace(chr(92),'/').split('/')[-3]; ds=d['dataset']
    cand={c['label']:c['retention_labels_all'] for c in d['candidates']}
    for r in d['results']:
        if r['reference']!='labels': continue
        for o in r['outcomes']:
            s=o['selected']
            if s is None: continue
            rows.append(dict(model=model,ds=ds,rule=r['rule'],target=r['target'],seed=o['seed'],sel=s,comp=o['compression'],cal=o['calibration'],hold=o['holdout'],allr=cand.get(s,np.nan)))
df=pd.DataFrame(rows)
df['slack']=(df.allr-df.target)*100
df['miss']=df.hold<df.target
df['short']=np.where(df.miss,(df.target-df.hold)*100,0)
df['merge']=df.sel.str.contains('merge')
print(len(df),df.allr.isna().sum(), df.rule.value_counts().to_dict())
for rule,g in df.groupby('rule'):
    m=g[g.miss]
    print(rule,len(g),'miss%',round(g.miss.mean()*100,2),'short mean',round(m.short.mean(),2) if len(m) else 0,'p95',round(np.percentile(m.short,95),2) if len(m) else 0,'max',round(g.short.max(),2),'slack<=0',round((g.slack<=0).mean()*100,1),'calwin',round(((g.cal-g.hold)*100).mean(),2))
bins=[-99,0,0.5,1,2,5,99]
df['bin']=pd.cut(df.slack,bins,right=True)
print(df.groupby('bin',observed=True).miss.agg(['mean','size']))
# merge vs codec for default
def cb(g,B=2000):
    rng=np.random.default_rng(0); ms=g.model.unique()
    sub={m:g[g.model==m] for m in ms}
    ds=[]
    for _ in range(B):
        pick=rng.choice(ms,len(ms)); x=pd.concat([sub[m] for m in pick])
        a=x[x['merge']].miss; b=x[~x['merge']].miss
        if len(a) and len(b): ds.append(a.mean()-b.mean())
    return np.percentile(ds,[2.5,97.5])
for rule in ['calibrate() default','lower 0.999','lower 0.975']:
    g=df[df.rule==rule]
    a=g[g['merge']].miss.mean();b=g[~g['merge']].miss.mean()
    print(rule,'merge',round(a*100,1),'codec',round(b*100,1),'n merge',g['merge'].sum(),cb(g)*100)
    # stratified by slack bin
    for bn,h in g.groupby('bin',observed=True):
        if h['merge'].any() and (~h['merge']).any(): print('   ',bn,len(h),round(h[h['merge']].miss.mean()*100,1),round(h[~h['merge']].miss.mean()*100,1),h['merge'].sum())
# robustness: exclude point rule from pooled, slack bins per rule
print(df[df.rule!='point'].groupby('bin',observed=True).miss.agg(['mean','size']))
# model clustering: per-model miss rates default
g=df[df.rule=='calibrate() default']; print(g.groupby('model').miss.mean().sort_values().tail(6))
print(g.groupby('ds').miss.mean().sort_values().tail(3))
# drop sibling/extreme: drop top miss model and recompute merge diff
top=g.groupby('model').miss.mean().idxmax(); h=g[g.model!=top]; print('drop',top,h[h['merge']].miss.mean()*100,h[~h['merge']].miss.mean()*100)
# slack<=0 share per rule
print(df.assign(le0=df.slack<=0).groupby('rule').le0.mean()*100)
# slack bins with model-cluster bootstrap CI for first two bins
rng=np.random.default_rng(1)
ms=df.model.unique(); gb={m:x for m,x in df.groupby('model')}
res={b:[] for b in df.bin.cat.categories}
for _ in range(500):
    x=pd.concat([gb[m] for m in rng.choice(ms,len(ms))])
    t=x.groupby('bin',observed=False).miss.mean()
    for b in res: res[b].append(t[b])
for b,v in res.items(): print(b,np.nanpercentile(v,[2.5,97.5]).round(3))
