import json,glob,numpy as np,collections
from scipy.stats import wilcoxon,spearmanr
W=collections.defaultdict(dict)
for f in glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json'):
    j=json.load(open(f));ds=j['dataset']
    if 'scifact' in ds.lower(): continue
    m=f.replace(chr(92),'/').split('/')[-3]
    W[(m,'D' if 'docvqa' in ds else 'I')]={r['label']:r['retention:ndcg@5'] if 'retention:ndcg@5' in r else None for r in j['rows']}
    W[(m,'D' if 'docvqa' in ds else 'I','b')]=j['baseline']['ndcg@5']
models=sorted({k[0] for k in W if (k[0],'D') in W and (k[0],'I') in W})
print(len(models))
labs=collections.Counter(l for m in models for l in W[(m,'D')] if l in W[(m,'I')])
print(len(labs), collections.Counter(labs.values()).most_common(5))
res=[]
rng=np.random.default_rng(1)
for l,c in labs.items():
    if c<30:continue
    ms=[m for m in models if W[(m,'D')].get(l) is not None and W[(m,'I')].get(l) is not None]
    x=np.array([W[(m,'D')][l]-W[(m,'I')][l] for m in ms])
    if np.all(x==0):continue
    p=wilcoxon(x).pvalue
    res.append((l,len(x),x.mean(),(x<0).mean(),p))
n=len(res);print('n pipelines',n,'bonf sig',sum(r[4]*n<.05 for r in res),'neg',sum(r[2]<0 for r in res))
for r in sorted(res,key=lambda r:r[2])[:6]:print(r)
# baseline confound: per-model correlation of diff for binary with baseline docvqa
bd=np.array([W[(m,'D','b')] for m in models]);bi=np.array([W[(m,'I','b')] for m in models])
for l in ['binary']:
    x=np.array([W[(m,'D')][l]-W[(m,'I')][l] for m in models])
    print('spearman diff vs docvqa baseline',spearmanr(x,bd),'vs infovqa baseline',spearmanr(x,bi))
    # absolute-ndcg loss difference
print('---low n')
out=[]
for l,c in labs.items():
    if c>=30 or c<10:continue
    ms=[m for m in models if W[(m,'D')].get(l) is not None and W[(m,'I')].get(l) is not None]
    x=np.array([W[(m,'D')][l]-W[(m,'I')][l] for m in ms])
    if np.all(x==0):continue
    out.append((l,len(x),round(x.mean(),4),round((x<0).mean(),2),wilcoxon(x).pvalue))
for o in sorted(out,key=lambda o:o[2])[:12]:print(o)
print(len(out),sum(o[2]<0 for o in out))
