import json,numpy as np
from scipy.stats import wilcoxon
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
def L(o): return json.load(open(R+o+"/wide/colbertv2.0_scifact.json"))
a=L("colbert-ir--colbertv2.0"); b=L("lightonai--colbertv2.0")
for x in (a,b):
    print({k:v for k,v in x['baseline'].items()}); print(x['n_docs'],x['n_queries'],x.get('environment'))
ra={r['label']:r for r in a['rows']}; rb={r['label']:r for r in b['rows']}
sh=[l for l in ra if l in rb]; print(len(ra),len(rb),len(sh))
d=np.array([ra[l]['retention:ndcg@10']-rb[l]['retention:ndcg@10'] for l in sh])
print(d.mean(),d.std(ddof=1),(d<0).sum(),(d>0).sum(),(d==0).sum(),wilcoxon(d).pvalue)
ad=np.array([ra[l]['ndcg@10']-rb[l]['ndcg@10'] for l in sh]);print(ad.mean())
# CI overlap, by group
for g in sorted({ra[l]['group'] for l in sh}):
    ls=[l for l in sh if ra[l]['group']==g]; print(g,len(ls),np.mean([ra[l]['retention:ndcg@10']-rb[l]['retention:ndcg@10'] for l in ls]))
# excluding retention 1.0 label identity; CI width
w=np.mean([ra[l]['retention_ci:ndcg@10'][1]-ra[l]['retention_ci:ndcg@10'][0] for l in sh]);print('ci width',w)
for l in sh[:0]:pass
for k in ('int4(mean)',):
    for l in sh:
        if k in l and ra[l]['group']=='codec': print(l,ra[l]['compression_vs_float32'],rb[l]['compression_vs_float32'],ra[l]['bytes_per_doc'],rb[l]['bytes_per_doc'])
for o in ("colbert-ir--colbertv2.0","lightonai--colbertv2.0"):
    r=json.load(open(R+o+"/run.json")); print(o,json.dumps(r.get('spec')),json.dumps(r.get('check_retriever'))[:500])
hr=json.load(open("C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json"))
t=sum(h['downloads'] for h in hr);print(t,[ (h['id'],h['downloads']) for h in hr if 'colbertv2' in h['id']], max(hr,key=lambda h:h['downloads'])['id'],len(hr))
