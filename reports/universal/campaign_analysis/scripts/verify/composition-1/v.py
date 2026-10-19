import os,json,glob,numpy as np,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
recs=[]
for f in sorted(glob.glob(R+'*/wide/*.json')):
    d=json.load(open(f)); m=os.path.basename(os.path.dirname(os.path.dirname(f)))
    ret={r['label']:r['retention:ndcg@5'] for r in d['rows']}
    ci={r['label']:r['retention_ci:ndcg@5'] for r in d['rows']}
    for l,v in ret.items():
        parts=l.split(' > ')
        if len(parts)<2: continue
        if all(p in ret for p in parts):
            recs.append(dict(m=m,ds=d['dataset'],l=l,n=len(parts),act=v,ci=ci[l],parts=[ret[p] for p in parts],f16='float16' in parts))
print(len(recs),len({r['m'] for r in recs}))
def stats(rs,name):
    if not rs: return
    a=np.array([r['act'] for r in rs]);p=np.array([np.prod(r['parts']) for r in rs]);ad=np.array([1-sum(1-x for x in r['parts']) for r in rs]);mn=np.array([min(r['parts']) for r in rs]);fs=np.array([r['parts'][0] for r in rs])
    e=p-a
    print(name,'n',len(rs),'models',len({r['m'] for r in rs}),'MAEprod %.4f med %.4f bias(pred-act) %.4f within.02 %.3f worst %.3f | add %.4f min %.4f first %.4f'%(abs(e).mean(),np.median(abs(e)),e.mean(),(abs(e)<.02).mean(),e.min(),abs(ad-a).mean(),abs(mn-a).mean(),abs(fs-a).mean()))
    return e
nf=[r for r in recs if not r['f16']]
e=stats(nf,'nof16'); stats(recs,'all'); stats([r for r in recs if r['f16']],'f16')
stats([r for r in nf if r['n']==2],'2stage');stats([r for r in nf if r['n']==3],'3stage')
# per model-level bootstrap
ms=sorted({r['m'] for r in nf}); by=collections.defaultdict(list)
for r in nf: by[r['m']].append(np.prod(r['parts'])-r['act'])
rng=np.random.default_rng(0)
b=[np.concatenate([by[ms[i]] for i in rng.integers(0,len(ms),len(ms))]).mean() for _ in range(2000)]
print('bias CI',np.percentile(b,[2.5,97.5]))
# per-model MAE
pm={m:np.mean(np.abs(by[m])) for m in ms}; s=sorted(pm.items(),key=lambda x:-x[1]); print(s[:6]); print('median model MAE',np.median(list(pm.values())))
# drop top3 models
for k in (1,3):
    top={m for m,_ in s[:k]}; stats([r for r in nf if r['m'] not in top],'drop%d'%k)
# regime: split by min part retention
for lo,hi in [(0,.8),(.8,.95),(.95,2)]:
    stats([r for r in nf if lo<=min(r['parts'])<hi],'minpart[%s,%s)'%(lo,hi))
stats([r for r in nf if ('scifact' in r['ds'])],'scifact'); stats([r for r in nf if ('scifact' not in r['ds'])],'image')
# naive: constant prediction = min vs mean-per-part baseline: predict act=1
a=np.array([r['act'] for r in nf]); print('predict 1.0 MAE',np.abs(1-a).mean(), 'mean loss',(1-a).mean())
# worst
for r in sorted(nf,key=lambda r:np.prod(r['parts'])-r['act'])[:3]+sorted(nf,key=lambda r:np.prod(r['parts'])-r['act'])[-3:]: print(r['m'],r['l'],r['act'],r['parts'])
# CI coverage of no-f16 controls
c=[r for r in recs if r['f16']]
print('f16 ctrl inside CI',np.mean([r['ci'][0]<=np.prod(r['parts'])<=r['ci'][1] for r in c]))
