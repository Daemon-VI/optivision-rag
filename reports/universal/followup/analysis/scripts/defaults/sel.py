import json,glob,collections,numpy as np,re
base='reports/universal/followup/'
def modality(path,d):
    t=path.lower()
    return 'image' if any(k in t for k in ['docvqa','infovqa','arxivqa','tabfquad','f4_','f5_','f6_','f7_']) else 'text'
units=[]
for f in glob.glob(base+'f[1-7]*/*/selection_rules/*.json'):
    if 'f3b' in f: continue
    units.append(f)
print(len(units))
agg=collections.defaultdict(list); pick=collections.defaultdict(collections.Counter)
for f in units:
    d=json.load(open(f)); m=modality(f.replace(chr(92),'/'),d)
    for r in d['results']:
        if r['reference']!='labels': continue
        agg[(m,r['rule'],r['target'])].append((r['met'],r['compression_median'],r['splits'],r['holdout_min']))
        if r['rule']=='calibrate() default':
            for o in r['outcomes']: pick[(m,r['target'])][o['selected']]+=1
print('modality rule target nunits met_mean met_min compmedian_mean(of units)')
for k in sorted(agg):
    v=np.array([x[:2] for x in agg[k]])
    print(k,len(v),round(v[:,0].mean(),3),round(v[:,0].min(),2),round(np.mean(v[:,1]),2),round(np.median(v[:,1]),2))
# paired compression default vs lower0.999 per unit
print('--- paired default vs lower 0.999')
for m in ['text','image']:
  for t in [0.95,0.97,0.99]:
    a={};b={}
    for f in units:
        if modality(f.replace(chr(92),'/'),None)!=m: continue
        d=json.load(open(f))
        for r in d['results']:
            if r['reference']=='labels' and r['target']==t:
                if r['rule']=='calibrate() default': a[f]=(r['compression_median'],r['met'])
                if r['rule']=='lower 0.999': b[f]=(r['compression_median'],r['met'])
    ks=list(a); ca=np.array([a[k][0] for k in ks]); cb=np.array([b[k][0] for k in ks])
    print(m,t,len(ks),'default comp mean',ca.mean().round(2),'lower999',cb.mean().round(2),'ratio gm',np.exp(np.mean(np.log(cb/ca))).round(2),'units l999 higher',(cb>ca*1.05).sum(),'lower',(cb<ca/1.05).sum())
print('--- picks')
for k in sorted(pick):
    print(k,sum(pick[k].values()),pick[k].most_common(8))
