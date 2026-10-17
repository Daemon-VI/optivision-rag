import json, itertools, numpy as np
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D={'doc':'vidore/docvqa_test_subsampled','info':'vidore/infovqa_test_subsampled'}
def prof(m,ds):
    e=d[m][ds]; return {r['label']:r['retention'] for r in e['rows']}, e['baseline']
def cmp(a,b,verbose=True):
    out=[]
    for k,ds in D.items():
        pa,ba=prof(a,ds);pb,bb=prof(b,ds)
        sh=sorted(set(pa)&set(pb)); x=np.array([pa[l]-pb[l] for l in sh])
        out.append((k,len(sh),round(np.abs(x).mean(),4),round(x.mean(),4),round(ba-bb,5)))
    print(a.split('/')[-1],'|',b.split('/')[-1],out)
P=[('tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'),
('vidore/colpali-v1.2','vidore/colpali-v1.2-merged'),('vidore/colpali-v1.3','vidore/colpali-v1.3-merged'),
('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),
('Metric-AI/ColQwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0'),
('vidore/colpali-v1.2','vidore/colpali-v1.3'),('vidore/colpali-v1.1','vidore/colpali-v1.2'),('vidore/colqwen2.5-v0.1','vidore/colqwen2.5-v0.2'),
('TomoroAI/tomoro-colqwen3-embed-4b','TomoroAI/tomoro-colqwen3-embed-8b')]
for a,b in P: cmp(a,b)
# all pairs
for k,ds in D.items():
    ms=[m for m in d if ds in d[m]]; v=[]
    for a,b in itertools.combinations(ms,2):
        pa,_=prof(a,ds);pb,_=prof(b,ds);sh=set(pa)&set(pb)
        if len(sh)>=20: v.append(np.mean([abs(pa[l]-pb[l]) for l in sh]))
    v=np.array(v);print(k,len(v),np.median(v),np.percentile(v,5),(v<0.005).sum())
# baselines
for ds in D.values():
  for m in ['TomoroAI/tomoro-colqwen3-embed-8b','vultr/VultronRetrieverCore-Qwen3.5-4.5B']: print(m,d[m][ds]['baseline'])
# noise: CI half-width typical
r=d['vidore/colpali-v1.2'][D['doc']]['rows']; print([ (x['label'],x['retention_interval']) for x in r[:3]])
for m in ['vidore/colpali-v1.2','vidore/colpali-v1.2-merged']:
  print(m,[(k,d[m][D[k]]['n_docs'],d[m][D[k]]['baseline'],d[m][D[k]]['note'][:100]) for k in D])
