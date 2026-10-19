import json,itertools,numpy as np
from scipy import stats
from optivision.adapters import MODELS
rng=np.random.default_rng(0)
m=json.load(open('src/optivision/presets/measured.json'))['models']
info={x.model_id:x for x in (MODELS.values() if isinstance(MODELS,dict) else MODELS)}
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
ids=[k for k in m if info[k].modality=='image' and info[k].dimension==128 and D in m[k] and I in m[k]]
print('dim128 image n',len(ids))
def ret(k,ds,lab):
    for r in m[k][ds]['rows']:
        if r['label']==lab: return r['retention'],r['retention_interval']
    return None
H='hierarchical_merge(0.25)'
sets={'A8':['int8(per_vector)','int4(mean)','binary','lloyd2(7)',H,H+' > int4(mean)',H+' > binary',H+' > lloyd2(7)'],
      'B7':['int4(mean)','binary','lloyd2(7)',H,H+' > int4(mean)',H+' > binary',H+' > lloyd2(7)']}
fam={k:info[k].family for k in ids}
for name,labs in sets.items():
    F={}
    for ds in (D,I):
        for k in ids:
            v=[ret(k,ds,l) for l in labs]
            F[k,ds]=None if any(x is None for x in v) else np.array([x[0] for x in v])
    ok=[k for k in ids if F[k,D] is not None and F[k,I] is not None]
    print(name,'models with all labels',len(ok))
    def frag(a): return np.mean(1-np.minimum(a,1))   # mean shortfall
    def frag2(a): return 1-np.min(a)
    for fn in (frag,frag2):
        x=np.array([fn(F[k,D]) for k in ok]);y=np.array([fn(F[k,I]) for k in ok])
        rho=stats.spearmanr(x,y)[0]
        bs=[]
        for _ in range(4000):
            ix=rng.integers(0,len(ok),len(ok)); bs.append(stats.spearmanr(x[ix],y[ix])[0])
        print(fn.__name__,'rho %.3f CI [%.2f,%.2f]'%(rho,*np.nanpercentile(bs,[2.5,97.5])),'pearson %.3f'%stats.pearsonr(x,y)[0])
        # leave-one-model-out range
        lo=[stats.spearmanr(np.delete(x,i),np.delete(y,i))[0] for i in range(len(ok))]
        print('  LOO range %.2f..%.2f'%(min(lo),max(lo)))
        # one per family (mean within family)
        fs=sorted(set(fam[k] for k in ok))
        xf=np.array([np.mean([x[i] for i,k in enumerate(ok) if fam[k]==f]) for f in fs]);yf=np.array([np.mean([y[i] for i,k in enumerate(ok) if fam[k]==f]) for f in fs])
        print('  family-mean rho %.2f n=%d'%(stats.spearmanr(xf,yf)[0],len(fs)))
        # permutation family eta^2
        def eta(vals,g):
            gm=vals.mean(); ss=sum(len(vals[g==f])*(vals[g==f].mean()-gm)**2 for f in set(g)); return ss/((vals-gm)**2).sum()
        g=np.array([fam[k] for k in ok])
        for nm,v in (('D',x),('I',y),('avg',(x+y)/2)):
            e=eta(v,g); p=np.mean([eta(rng.permutation(v),g)>=e for _ in range(5000)])
            print('  eta2 %s %.2f p=%.3f'%(nm,e,p))
    print(sorted(set(fam[k] for k in ok)),{f:sum(fam[k]==f for k in ok) for f in set(fam[k] for k in ok)})
    # worst models
    x=np.array([frag(F[k,D]) for k in ok]);y=np.array([frag(F[k,I]) for k in ok])
    for i in np.argsort(-x)[:6]: print('  D',ok[i],round(x[i],3),round(y[i],3))
    # float16 baseline
    f16=[i for i,k in enumerate(ok) if 'float16' in (info[k].evidence or '').lower()]
    print('  f16 models',[ok[i] for i in f16])
    if f16 and len(f16)<len(ok):
        o=[i for i in range(len(ok)) if i not in f16]
        for arr,nm in ((x,'D'),(y,'I')): print('  MWU',nm,stats.mannwhitneyu(arr[f16],arr[o]).pvalue, arr[f16].mean(),arr[o].mean())
