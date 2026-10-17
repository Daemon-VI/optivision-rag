import json,glob,numpy as np,collections,os
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
rng=np.random.default_rng(0)
files=sorted(glob.glob(R+'*/selection_rules/*.json'))
print(len(files))
cells={}  # (ref,model,ds,rule,target)->(met,comp, mets)
models=set()
for f in files:
    m=os.path.basename(os.path.dirname(os.path.dirname(f)))
    d=json.load(open(f)); models.add(m)
    for r in d['results']:
        oc=r['outcomes']
        cells[(r['reference'],m,d['dataset'],r['rule'],r['target'])]=(r['met'],r['compression_median'],r['met_labels'],len(oc))
print(len(models))
rules=sorted(set(k[3] for k in cells))
for ref in ['labels','baseline@1']:
  print('REF',ref)
  ms=sorted(set(k[1] for k in cells if k[0]==ref))
  print(len(ms),'models', len([k for k in cells if k[0]==ref and k[3]==rules[0]]),'cells')
  for rule in rules:
    pm={}
    for m in ms:
        v=[cells[k][0] for k in cells if k[0]==ref and k[1]==m and k[3]==rule]
        pm[m]=np.mean(v)
    arr=np.array(list(pm.values()))
    # pooled over cells, cluster bootstrap
    mm={m:[cells[k][0] for k in cells if k[0]==ref and k[1]==m and k[3]==rule] for m in ms}
    cc={m:[np.log(cells[k][1]) for k in cells if k[0]==ref and k[1]==m and k[3]==rule] for m in ms}
    def stat(idx):
        a=np.concatenate([mm[ms[i]] for i in idx]);c=np.concatenate([cc[ms[i]] for i in idx]);return a.mean(),np.exp(c.mean())
    pt=stat(range(len(ms)))
    bs=np.array([stat(rng.integers(0,len(ms),len(ms))) for _ in range(1000)])
    lo,hi=np.percentile(bs[:,0],[2.5,97.5])
    bt={}
    for t in (0.95,0.97,0.99):
        v=[cells[k][0] for k in cells if k[0]==ref and k[3]==rule and k[4]==t]; bt[t]=round(np.mean(v),3)
    print(f'{rule:30s} met {pt[0]:.3f} [{lo:.3f},{hi:.3f}] gm {pt[1]:.1f} below95 {(arr<0.95).sum()} min {arr.min():.3f} {min(pm,key=pm.get)} per-target {bt}')
