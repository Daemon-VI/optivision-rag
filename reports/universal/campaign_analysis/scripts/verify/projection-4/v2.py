import json,glob,numpy as np,collections
from optivision.adapters import MODELS
m=json.load(open('src/optivision/presets/measured.json'))['models']
c=collections.Counter()
for mid,ds in m.items():
    for d,x in ds.items():
        for r in x['rows']:
            if 'project(32' in r['label'] and 'binary' not in r['label'] and 'int' not in r['label'] and 'float' not in r['label']: c[r['label']]+=1
print(c)
mid=[k for k in m if MODELS[k].modality=='text'][0]
d=list(m[mid])[0]
print(mid,[ (r['label'],round(r['retention'],3)) for r in m[mid][d]['rows'] if 'project(32' in r['label'] or r['label'].startswith('hierarchical_merge') and 'project' not in r['label']][:30])
print(list(MODELS.values())[0])
