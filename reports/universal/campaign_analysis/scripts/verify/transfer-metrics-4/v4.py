import json,numpy as np,time
from scipy.stats import rankdata
rng=np.random.default_rng(1)
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
    a=rankdata(a);b=rankdata(b);a=a-a.mean();b=b-b.mean()
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())
BANDS=[(1,8),(8,32),(32,128),(128,1e12)]
def run(ds,M,excl,tag,NP=300,boot=False):
    names=[m for m in M if not excl(m)];b=np.array([M[m][0] for m in names]);n=len(names)
    out=[]
    for lo,hi in BANDS:
        pk={}
        for i,m in enumerate(names):
            for k,(r,c) in M[m][1].items():
                if lo<=c<hi: pk.setdefault(k,[]).append((i,r))
        P=[(np.array([x[0] for x in v]),np.array([x[1] for x in v])) for v in pk.values() if len(v)>=10]
        if not P: out.append('none');continue
        f=lambda bb:np.mean([rho(bb[i],r) for i,r in P])
        o=f(b);pv=np.array([f(rng.permutation(b)) for _ in range(NP)])
        p=(np.sum(np.abs(pv)>=abs(o))+1)/(NP+1)
        s=f'{lo}-{hi:.0f}: k={len(P)} rho={o:.2f} p={p:.3f}(null sd {pv.std():.2f})'
        if boot:
            bs=[]
            for _ in range(100):
                sm=rng.integers(0,n,n);rs=[]
                for i,r in P:
                    pos={m:j for j,m in enumerate(i)};sel=[pos[x] for x in sm if x in pos]
                    rs.append(rho(b[i][sel],r[sel]))
                bs.append(np.mean(rs))
            s+=f' CI[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]'
        out.append(s)
    print('==',ds,tag,n);print('\n'.join(out),flush=True)
for ds in ['scifact']:
    M=load(ds);print(ds,len(M),'f16',sum(v[2] for v in M.values()),'fams',sorted({fam.get(m,'?') for m in M}),flush=True)
    bl=sorted(M,key=lambda m:M[m][0])
    print('lowest',[(round(M[m][0],3),m.split('/')[-1][:20]) for m in bl[:4]])
    run(ds,M,lambda m:False,'all',boot=True)
    run(ds,M,lambda m:m in bl[:2],'drop 2 lowest')
    run(ds,M,lambda m:m in (bl[0],bl[-1]),'drop min,max')
    run(ds,M,lambda m:M[m][2],'drop f16')
    seen={}
    for m in sorted(M,key=lambda m:-M[m][0]): seen.setdefault(fam.get(m,m),m)
    run(ds,M,lambda m:m not in seen.values(),'one/family')
print('end',flush=True)
