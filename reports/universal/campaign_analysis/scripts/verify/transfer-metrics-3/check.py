import json,glob,os,numpy as np
from scipy.stats import wilcoxon
rng=np.random.default_rng(0)
M=['recall@1','ndcg@5','ndcg@10','recall@5']
base="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign"
data={}
for f in glob.glob(base+"/*/wide/*.json"):
    d=json.load(open(f)); mod=f.replace("\\","/").split("/")[-3]
    ds='SciFact' if 'scifact' in d['dataset'].lower() else ('DocVQA' if 'docvqa' in d['dataset'].lower() else 'InfoVQA')
    R=np.array([[r['retention:'+m] for m in M] for r in d['rows']])
    G=[r['group'] for r in d['rows']]
    data.setdefault(ds,{})[mod]=(R,G,np.array([r['compression_vs_float32'] for r in d['rows']]),d['baseline']['encoder'].get('dtype'))
def boot(v):
    v=np.array(v);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(2000)];return v.mean(),np.percentile(b,[2.5,97.5])
ntest=0
for ds,mods in data.items():
    print("==",ds,len(mods),"models")
    for sel in ['nonsat','all']:
        per={}
        for k,(R,G,C,dt) in mods.items():
            m=R.min(1)<0.97 if sel=='nonsat' else np.ones(len(R),bool)
            if m.sum()>=3: per[k]=R[m].mean(0)
        P=np.array(list(per.values()))
        full=np.mean([(p[0]<p[1]<p[2]<p[3]) for p in P])
        print(sel,"n_models",len(P),"mean ret",P.mean(0).round(4),"full order share",round(full,2))
        for a,b in [(0,1),(3,1),(1,2),(3,2),(0,3)]:
            d=P[:,a]-P[:,b]; mu,ci=boot(d); ntest+=1
            print(f"  {M[a]}-{M[b]}: {mu:+.3f} CI({ci[0]:+.3f},{ci[1]:+.3f}) pos {int((d>0).sum())}/{len(d)} wilcoxon p={wilcoxon(d).pvalue:.1e}")
    if True:
        # argmin row share, nonsat
        am=[];
        for k,(R,G,C,dt) in mods.items():
            m=R.min(1)<0.97; am+= list((R[m].argmin(1)))
        print(" recall@1 lowest share(nonsat rows pooled):",round(np.mean(np.array(am)==0),3))
        # leave-out extreme + fp16 exclusion
        P={k:(R[R.min(1)<0.97].mean(0)) for k,(R,G,C,dt) in mods.items() if (R.min(1)<0.97).sum()>=3}
        d={k:v[0]-v[1] for k,v in P.items()}
        ex=min(d,key=d.get); print(" most extreme",ex,round(d[ex],3))
        dd=np.array([v for k,v in d.items() if k!=ex]);print(" w/o extreme",dd.mean().round(3))
        nf=np.array([d[k] for k in d if 'float16' not in str(mods[k][3])]);print(" non-fp16 models",len(nf),nf.mean().round(3),"fp16",len(d)-len(nf))
        # by compression
        for lo,hi in [(0,4),(4,16),(16,64),(64,256),(256,1e9)]:
            v=[]
            for k,(R,G,C,dt) in mods.items():
                m=(R.min(1)<0.97)&(C>=lo)&(C<hi)
                if m.sum()>=3:v.append((R[m,0]-R[m,1]).mean())
            if v:print(f"  comp[{lo},{hi}) n={len(v)} r1-n5 {np.mean(v):+.3f}")
        for g in ['codec','token','token+dim','token+dim+codec']:
            v=[]
            for k,(R,G,C,dt) in mods.items():
                m=(R.min(1)<0.97)&(np.array(G)==g)
                if m.sum()>=3:v.append((R[m,0]-R[m,1]).mean())
            if v:print(f"  group {g} n={len(v)} {np.mean(v):+.3f}")
print("tests",ntest)
