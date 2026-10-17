import json,glob,numpy as np
from scipy import stats
R={}
for f in glob.glob("C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json"):
    j=json.load(open(f)); L={}
    for r in j['rows']:
        L[r['label']]=r['retention:ndcg@5'] if 'retention:ndcg@5' in r else None
    if all(k in L for k in ['int4','binary','int4(mean)','int8(per_vector)']):
        R[f.split('campaign')[1][:50]+'|'+j['dataset']]=(L['int4'],L['binary'],L['int4(mean)'],L['int8(per_vector)'])
print(len(R))
for k,v in sorted(R.items(),key=lambda kv:kv[1][0])[:12]: print(k,["%.3f"%x for x in v])
txt={k:v for k,v in R.items() if 'scifact' in k}
print(len(txt))
for k,v in sorted(txt.items(),key=lambda kv:kv[1][0]): print(k,["%.3f"%x for x in v])
a=np.array([[1-v[0],1-v[1]] for v in txt.values()])
print(stats.spearmanr(*a.T),stats.pearsonr(*a.T))
