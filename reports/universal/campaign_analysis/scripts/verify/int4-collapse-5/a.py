import json
from optivision.adapters import MODELS
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
for mid,ds in m.items():
    info=MODELS.get(mid)
    if info is None or info.modality!='text': continue
    for d,v in ds.items():
        r={x['label']:x for x in v['rows']}
        g=lambda k:(round(r[k]['retention'],4) if k in r else None)
        i4=r.get('int4')
        print(mid,info.dimension,v['n_docs'],g('int4'),i4 and i4['retention_interval'],g('int4(mean)'),g('binary'))
