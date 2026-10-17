import json,sys,numpy as np
from scipy.stats import spearmanr,rankdata
rng=np.random.default_rng(0)
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
from optivision.adapters import MODELS
fam={m.model_id:m.family for m in MODELS.values()} if isinstance(MODELS,dict) else {m.model_id:m.family for m in MODELS}
def key(r): return json.dumps(r['pipeline'],sort_keys=True)
def load(ds):
    out={}
    for m,v in d.items():
        for k in v:
            if ds in k:
                out[m]=(v[k]['baseline'],{key(r):(r['retention'],r['compression']) for r in v[k]['rows']},'float16' in v[k].get('note',''))
    return out
def sp(a,b):
    return spearmanr(a,b)[0]
def stat(M,names,lo,hi,minm=10,perm=None):
    # per pipeline rho
    pk={}
    for m in names:
        for k,(r,c) in M[m][1].items():
            if lo<=c<hi: pk.setdefault(k,[]).append(m)
    ks=[k for k,v in pk.items() if len(v)>=minm]
    return ks,pk
def meanrho(M,names,ks,pk,bperm=None):
    rs=[]
    for k in ks:
        ms=[m for m in pk[k] if m in names]
        if len(ms)<8: continue
        b=np.array([M[m][0] for m in ms]); 
        if bperm is not None: b=np.array([bperm[m] for m in ms])
        r=np.array([M[m][1][k][0] for m in ms])
        rs.append(sp(b,r))
    return np.mean(rs) if rs else np.nan,len(rs)
def run(ds,excl=lambda m:False,tag=''):
    M=load(ds); names=[m for m in M if not excl(m)]
    print('==',ds,tag,len(names),'models')
    for lo,hi in[(1,8),(8,32),(32,128),(128,1e9)]:
        ks,pk=stat(M,names,lo,hi)
        # compression of pipeline may vary slightly across models; use per-model
        obs,n=meanrho(M,names,ks,pk)
        vals=[];
        for _ in range(300):
            p=rng.permutation(len(names)); bp={m:M[names[i]][0] for m,i in zip(names,p)}
            vals.append(meanrho(M,names,ks,pk,bp)[0])
        pv=(np.sum(np.abs(vals)>=abs(obs))+1)/(len(vals)+1)
        bs=[]
        for _ in range(300):
            s=list(rng.choice(names,len(names)));
            # bootstrapped model resample: duplicates; approximate by jitter-free
            rs=[]
            for k in ks:
                ms=[m for m in s if k in M[m][1]]
                if len(ms)<8: continue
                rs.append(sp([M[m][0] for m in ms],[M[m][1][k][0] for m in ms]))
            bs.append(np.nanmean(rs))
        print(f'{lo}-{hi}: pipes={n} rho={obs:.2f} p={pv:.3f} CI={np.nanpercentile(bs,2.5):.2f},{np.nanpercentile(bs,97.5):.2f}')
    return M
for ds in['docvqa','infovqa']:
    M=load(ds); print(ds,len(M),'f16',sum(v[2] for v in M.values()))
    run(ds)
    # remove extremes
    bl=sorted(M,key=lambda m:M[m][0])
    run(ds,lambda m:m in bl[:2],'drop 2 lowest-baseline')
    run(ds,lambda m:M[m][2],'drop f16')
    # one per family (best)
    seen={}
    for m in sorted(M,key=lambda m:-M[m][0]): seen.setdefault(fam.get(m,m),m)
    run(ds,lambda m:m not in seen.values(),'one per family')
