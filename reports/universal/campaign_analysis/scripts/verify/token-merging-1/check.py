import json,re,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
K=[0.5,0.33,0.25,0.1]
rng=np.random.default_rng(0)
tab={}
for m,dd in d.items():
    for ds,v in dd.items():
        base=[r for r in v['rows'] if r['label']=='float32']
        b=base[0]['vectors_per_doc'] if base else None
        row={}
        for r in v['rows']:
            mm=re.fullmatch(r'hierarchical_merge\(([\d.]+)\)',r['label'])
            if mm and float(mm.group(1)) in K:
                row[float(mm.group(1))]=(r['retention'],r['retention_interval'][1],r['vectors_per_doc']/b if b else None)
        if len(row)==4: tab.setdefault(ds,{})[m]=row
fam=lambda m:m.split('/')[0]
for ds,t in tab.items():
    ms=list(t); print(ds,len(ms))
    A=np.array([[t[m][k][0] for k in K] for m in ms])
    print(' mean',A.mean(0).round(4),'median',np.median(A,0).round(4))
    fr=np.array([[t[m][k][2] for k in K] for m in ms]); print(' actual keep mean',fr.mean(0).round3() if False else fr.mean(0).round(3))
    for a,b in [(0,2),(2,3)]:
        dl=(A[:,a]-A[:,b])*100
        bs=[rng.choice(dl,len(dl)).mean() for _ in range(5000)]
        print(' step',K[a],K[b],'%.2f'%dl.mean(),np.percentile(bs,[2.5,97.5]).round(2),'wilcoxon p=%.1e'%stats.wilcoxon(dl).pvalue)
    print(' share>=.99 at .1',(A[:,3]>=.99).mean().round(2),' CIub<.99:',sum(t[m][0.1][1]<.99 for m in ms),'/',len(ms))
    # exponent fit
    x=np.log2(1/np.array(K)); 
    def fit(A):
        L=1-A.mean(0); ps=np.linspace(0.5,4,351); best=None
        for p in ps:
            f=x**p; s=(f*L).sum()/(f*f).sum(); e=((L-s*f)**2).sum()
            if best is None or e<best[0]: best=(e,p)
        return best[1]
    ps=[fit(A[rng.integers(0,len(A),len(A))]) for _ in range(1000)]
    print(' p',fit(A),np.percentile(ps,[2.5,97.5]).round(2))
    # without most extreme model / sibling families (one per org)
    last=(A[:,2]-A[:,3]); i=last.argmax(); print(' max step model',ms[i],round(last[i]*100,2))
    keep=[j for j in range(len(ms)) if j!=i]
    B=A[keep]; print(' drop extreme step .25->.1: %.2f'%((B[:,2]-B[:,3]).mean()*100),'0.5->.25 %.2f'%((B[:,0]-B[:,2]).mean()*100), 'p',fit(B))
    seen={};idx=[]
    for j,m in enumerate(ms):
        if fam(m) not in seen: seen[fam(m)]=1; idx.append(j)
    C=A[idx]; print(' one per org n=',len(idx),'mean',C.mean(0).round(4),'p',fit(C))
    print(' n cells >2pp extra step:',((A[:,2]-A[:,3])-(A[:,0]-A[:,2])>0.02).sum())
