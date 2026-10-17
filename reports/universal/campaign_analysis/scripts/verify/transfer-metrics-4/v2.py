import json,numpy as np
from scipy.stats import rankdata
rng=np.random.default_rng(0)
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
from optivision.adapters import MODELS
fam={m.model_id:m.family for m in (MODELS.values() if isinstance(MODELS,dict) else MODELS)}
key=lambda r:json.dumps(r['pipeline'],sort_keys=True)
def load(ds):
    o={}
    for m,v in d.items():
        for k in v:
            if ds in k:
                o[m]=(v[k]['baseline'],{key(r):(r['retention'],r['compression']) for r in v[k]['rows']},'float16' in v[k].get('note',''))
    return o
def rho(a,b):
    a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean()
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-300)
BANDS=[(1,8),(8,32),(32,128),(128,1e9)]
def prep(M,names,lo,hi):
    pk={}
    for i,m in enumerate(names):
        for k,(r,c) in M[m][1].items():
            if lo<=c<hi: pk.setdefault(k,[]).append(i)
    ks=[k for k,v in pk.items() if len(v)>=10]
    return [(np.array(pk[k]),np.array([M[names[i]][1][k][0] for i in pk[k]])) for k in ks]
def mr(P,b):
    return np.mean([rho(b[i],r) for i,r in P])
def run(ds,excl=lambda m:False,tag='',M=None):
    M=M or load(ds);names=[m for m in M if not excl(m)];b=np.array([M[m][0] for m in names])
    print('==',ds,tag,len(names))
    for lo,hi in BANDS:
        P=prep(M,names,lo,hi)
        if not P: print(lo,hi,'none');continue
        o=mr(P,b);pv=[mr(P,rng.permutation(b)) for _ in range(300)]
        p=(np.sum(np.abs(pv)>=abs(o))+1)/1001
        bs=[]
        for _ in range(100):
            s=rng.integers(0,len(names),len(names));rs=[]
            for i,r in P:
                cnt=np.bincount(i,minlength=len(names));idx=np.repeat(np.arange(len(names)),cnt[np.arange(len(names))]*0+1) # placeholder
                # resample: for each drawn model present in this pipeline
                pos={m:j for j,m in enumerate(i)}
                sel=[pos[x] for x in s if x in pos]
                if len(sel)<8: continue
                rs.append(rho(b[i][sel],r[sel]))
            bs.append(np.mean(rs))
        print(f'sd0={np.std(pv):.3f} {lo}-{hi}: pipes={len(P)} rho={o:.2f} p={p:.3f} CI=[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]')
for ds in['docvqa','infovqa','scifact']:
    M=load(ds);print(ds,len(M),'f16',sum(v[2] for v in M.values()),'fams',len({fam.get(m) for m in M}))
    if not M: continue
    run(ds,M=M)
    bl=sorted(M,key=lambda m:M[m][0])
    run(ds,lambda m:m in bl[:2],'drop 2 lowest-baseline',M)
    run(ds,lambda m:m in bl[:1]+bl[-1:],'drop min&max',M)
    run(ds,lambda m:M[m][2],'drop f16',M)
    seen={}
    for m in sorted(M,key=lambda m:-M[m][0]): seen.setdefault(fam.get(m,m),m)
    run(ds,lambda m:m not in seen.values(),'one per family',M)
    print('baselines',[ (round(M[m][0],3),m.split('/')[-1][:18]) for m in bl[:5]])
