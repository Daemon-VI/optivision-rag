import json,collections
from optivision.adapters import MODELS
m=json.load(open('src/optivision/presets/measured.json'))['models']
print(len(m))
c=collections.Counter()
for mid,d in m.items():
    for ds,v in d.items():
        for r in v['rows']: c[r['label']]+=1
print(len(c)); print([k for k,_ in c.most_common(60)])
dims=collections.Counter()
for mid in m:
    mi=MODELS.get(mid) if isinstance(MODELS,dict) else None
    if mi is None:
        mi=[x for x in MODELS if x.model_id==mid]; mi=mi[0] if mi else None
    print(mid, list(m[mid].keys()), getattr(mi,'dimension',None), getattr(mi,'family',None), getattr(mi,'modality',None), (getattr(mi,'evidence','') or '')[:60])
