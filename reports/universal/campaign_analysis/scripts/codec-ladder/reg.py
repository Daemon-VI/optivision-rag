from optivision.adapters import MODELS
import json
out={}
for k,v in (MODELS.items() if isinstance(MODELS,dict) else [(m.model_id,m) for m in MODELS]):
    out[k]=dict(family=v.family,modality=v.modality,dimension=v.dimension,status=v.status,evidence=str(v.evidence)[:160],license=v.license)
json.dump(out,open('registry.json','w'),indent=1)
print(len(out)); 
import collections
print(collections.Counter(v['family'] for v in out.values()))
for k in list(out)[:3]: print(k,out[k])
