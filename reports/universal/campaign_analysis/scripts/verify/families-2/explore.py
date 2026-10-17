import json
from optivision.adapters import MODELS
m=json.load(open('src/optivision/presets/measured.json'))['models']
for k,v in m.items():
    info=MODELS.get(k) if hasattr(MODELS,'get') else None
    print(k, info.family if info else None, info.modality if info else None, info.dimension if info else None, list(v))
r=m[next(iter(m))]
d=next(iter(r.values()))
print(d['baseline'],d['note'])
print(sorted({x['label'] for x in d['rows']})[:80], len(d['rows']))
