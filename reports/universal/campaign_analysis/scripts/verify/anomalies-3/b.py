import json, itertools, numpy as np
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
ds='vidore/infovqa_test_subsampled'
ms=[m for m in d if ds in d[m]]
P=lambda m:{r['label']:r for r in d[m][ds]['rows']}
res=[]
for a,b in itertools.combinations(ms,2):
    pa,pb=P(a),P(b);sh=set(pa)&set(pb)
    if len(sh)>=20: res.append((np.mean([abs(pa[l]['retention']-pb[l]['retention']) for l in sh]),a,b,len(sh)))
res.sort()
for r in res[:12]: print(r)
# CI half-width for merged pair, and excluding near-1 rows
a,b='vidore/colpali-v1.2','vidore/colpali-v1.2-merged'
pa,pb=P(a),P(b);sh=sorted(set(pa)&set(pb))
hw=[(pa[l]['retention_interval'][1]-pa[l]['retention_interval'][0])/2 for l in sh]
print('median CI halfwidth',np.median(hw))
dif=np.array([abs(pa[l]['retention']-pb[l]['retention']) for l in sh]); ret=np.array([pa[l]['retention'] for l in sh])
print('rows ret>0.98',(ret>0.98).sum(),'mad those',dif[ret>0.98].mean(),'mad ret<0.9',dif[ret<0.9].mean() if (ret<0.9).any() else None, 'n<0.9',(ret<0.9).sum())
print('max diff',dif.max(), sh[dif.argmax()])
# tsystems pair
a,b='tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'
for dd in ['vidore/docvqa_test_subsampled',ds]:
  pa={r['label']:r for r in d[a][dd]['rows']};pb={r['label']:r for r in d[b][dd]['rows']}
  sh=sorted(pa);dif=np.array([abs(pa[l]['retention']-pb[l]['retention']) for l in sh]);ret=np.array([pa[l]['retention'] for l in sh])
  print(dd[:10],len(sh),'ret>0.98',(ret>0.98).sum(),dif[ret>0.98].mean(),'ret<0.9',(ret<0.9).sum(),dif[ret<0.9].mean(),'max',dif.max(),d[a][dd]['baseline'],d[b][dd]['baseline'])
