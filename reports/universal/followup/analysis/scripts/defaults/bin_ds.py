import json,collections,numpy as np
meta=json.load(open('reports/universal/followup/analysis/scripts/defaults/units.json'))
K='retention_ci:ndcg@5';P='retention:ndcg@5'
d=collections.defaultdict(list)
for f,org,img,fol,dim in meta:
    if not img or dim!=128: continue
    ds=[k for k in ['docvqa','infovqa','arxivqa','tabfquad'] if k in f.lower()][0]
    rows={r['label']:r for r in json.load(open(f))['rows']}
    d[ds].append((rows['binary'][K][0],rows['binary'][P],rows['int4(mean)'][K][0]))
for k,v in d.items():
    a=np.array(v); print(k,len(v),'binary safe',(a[:,0]>=.95).mean().round(2),'med pt',np.median(a[:,1]).round(3),'min pt',a[:,1].min().round(3),'int4mean safe',(a[:,2]>=.95).mean())
