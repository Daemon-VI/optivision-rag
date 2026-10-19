import json,numpy as np,collections
from scipy import stats
from optivision.adapters import MODELS
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
def get(mid,d,label):
    for r in m[mid][d]['rows']:
        if r['label']==label: return r['retention']
res={}
for mid in m:
    for d in m[mid]:
        k=d.split('/')[-1][:4]; res[(mid,k)]=(MODELS[mid].dimension,MODELS[mid].family,get(mid,d,'binary'),MODELS[mid].evidence)
res={k:v for k,v in res.items() if v[2] is not None}
# family listing for image
fam=collections.defaultdict(list)
for (mid,k),v in res.items():
    if k=='docv': fam[v[1]].append((v[0],mid.split('/')[-1]))
for f,v in fam.items(): print(f,sorted(v))
# float16 inference share
for (mid,k),v in res.items():
    if k=='docv' and 'float16' in v[3].lower(): print('f16',mid,v[0])
# text drop outliers
t=[(v[0],1-v[2],mid) for (mid,k),v in res.items() if k not in ('docv','info') and v[2] is not None]
t2=[x for x in t if x[0]==128]; print('text n128',len(t2))
print('text outliers',[x for x in t if x[0]!=128])
t3=[x for x in t]; print('spearman w/o 48',stats.spearmanr([x[0] for x in t if x[0]!=48],[x[1] for x in t if x[0]!=48]))
# image: PCA128 binary incr loss high-dim vs bin native; and within-128 vs rest, MWU
for k in ('docv','info'):
    a=[1-v[2] for (mid,kk),v in res.items() if kk==k and v[0]==128]
    b=[1-v[2] for (mid,kk),v in res.items() if kk==k and v[0]>128]
    print(k,'128 n',len(a),np.median(a),'>128 n',len(b),np.median(b),stats.mannwhitneyu(a,b))
    # excluding colqwen3 / colpali families: compare by family medians 
    # bootstrap diff of medians over models
    rng=np.random.default_rng(1);ds=[np.median(rng.choice(a,len(a)))-np.median(rng.choice(b,len(b))) for _ in range(5000)]
    print(' diff med CI',np.percentile(ds,[2.5,97.5]))
# PCA tests
for k in ('docv','info'):
    inc64=[];inc128=[];
    for (mid,kk),v in res.items():
        if kk!=k: continue
        dkey=[d for d in m[mid] if d.split('/')[-1][:4]==k][0]
        def inc(kd):
            b=get(mid,dkey,f'project({kd}) > binary'); f=get(mid,dkey,f'project({kd}) > float16')
            return None if b is None or f is None else f-b
        if v[0]==128:
            i=inc(64); 
            if i is not None: inc64.append((mid,i,1-v[2]))
        if v[0]>=320:
            i=inc(128)
            if i is not None: inc128.append((mid,i,1-v[2]))
    print(k,'n128->pca64',len(inc64))
    nat=np.array([1-get(mid,[d for d in m[mid] if d.split('/')[-1][:4]==k][0],'binary')*0 for mid,_,_ in inc64]) if 0 else None
    d=[i-(l_f) for mid,i,l_f in [(mid,i,0) for mid,i,l in inc64]]
    # native incremental loss = float16 vs binary native
    nat=[]
    for mid,i,l in inc64:
        dkey=[d for d in m[mid] if d.split('/')[-1][:4]==k][0]
        nat.append(get(mid,dkey,'float16')-get(mid,dkey,'binary'))
    diff=np.array([i for _,i,_ in inc64])-np.array(nat)
    print(' pca64 - native incr loss mean',diff.mean(),'wilcoxon',stats.wilcoxon(diff).pvalue,'frac worse',(diff>0).mean(), 'dropping top3',np.sort(diff)[:-3].mean())
    nat2=[]
    for mid,i,l in inc128:
        dkey=[d for d in m[mid] if d.split('/')[-1][:4]==k][0]
        nat2.append(get(mid,dkey,'float16')-get(mid,dkey,'binary'))
    diff2=np.array([i for _,i,_ in inc128])-np.array(nat2)
    print(' n',len(inc128),'hi-dim nat incr',np.mean(nat2),'pca128 incr',np.mean([i for _,i,_ in inc128]),'wilcoxon',stats.wilcoxon(diff2).pvalue)
    print(' hi-dim models', [(mid.split('/')[-1],round(float(i),4)) for mid,i,_ in inc128])
