from lomo import *
import pandas as pd
from scipy.stats import beta, rankdata
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
dimu=np.array([u['dim'] for u in U]); vpdu=np.array([u['vpd'] for u in U])
# model-level metadata
def trainmodels(m,strat):
    um=minfo[m]
    others=[p for p in models if p!=m and minfo[p]['cluster']!=um['cluster']]
    if strat=='global': return others
    if strat=='modality': return [p for p in others if minfo[p]['modality']==um['modality']]
    if strat=='samedim': 
        s=[p for p in others if minfo[p]['dim']==um['dim']]; return s
    if strat=='samedim_mod':
        return [p for p in others if minfo[p]['dim']==um['dim'] and minfo[p]['modality']==um['modality']]
    if strat=='family': return [p for p in others if minfo[p]['family']==um['family']]
    if strat=='dimband':  # dim bucket: <=96, 128, 320-640, >=2048
        b=lambda d: 0 if d<=96 else 1 if d==128 else 2 if d<=640 else 3
        return [p for p in others if b(minfo[p]['dim'])==b(um['dim'])]
    if strat=='dimband_mod':
        b=lambda d: 0 if d<=96 else 1 if d==128 else 2 if d<=640 else 3
        return [p for p in others if b(minfo[p]['dim'])==b(um['dim']) and minfo[p]['modality']==um['modality']]
    if strat.startswith('knn'):
        k=int(strat[3:]); return knn_models(m,others,k,False,True)
    if strat.startswith('dknn'):  # nearest in log2 dim only, same modality-agnostic
        k=int(strat[4:]); pool=others
        d=[abs(math.log2(minfo[p]['dim'])-math.log2(um['dim']))+0.01*abs(math.log2(minfo[p]['vpd'])-math.log2(um['vpd'])) for p in pool]
        o=np.argsort(d)[:k]; return [pool[i] for i in o]
STR=['global','modality','family','samedim','samedim_mod','dimband','dimband_mod','knn5','knn10','dknn5','dknn10']
PRI=4.0
def probs(m,ui,strat,t,rows_cache):
    ms=trainmodels(m,strat)
    rows=[i for p in ms for i in midx[p]]
    if not rows: return None,0
    r=R[rows]; av=~np.isnan(r)
    h=((r>=t)&av).sum(0); n=av.sum(0)
    # prior from global leave-cluster-out pass rates
    return (h,n),len(ms)
allrec=[]
glob_cache={}
for t in TARGETS:
  for m in models:
    others=[p for p in models if p!=m and minfo[p]['cluster']!=minfo[m]['cluster']]
    rows=[i for p in others for i in midx[p]]
    r=R[rows]; av=~np.isnan(r); gh=((r>=t)&av).sum(0); gn=av.sum(0)
    glob_cache[(t,m)]=(gh+0.5)/(gn+1)
for strat in STR:
  for t in TARGETS:
    for m in models:
        res,nm=probs(m,None,strat,t,None)
        if res is None: continue
        h,n=res; gp=glob_cache[(t,m)]
        for ui in midx[m]:
            y=R[ui]>=t
            ok=~np.isnan(R[ui])
            for j in np.where(ok)[0]:
                a=PRI*gp[j]; b=PRI*(1-gp[j])
                p=(h[j]+a)/(n[j]+PRI)
                lo=beta.ppf(0.1,h[j]+a,n[j]-h[j]+b)
                allrec.append((strat,t,m,ui,j,p,lo,int(y[j]),n[j],nm))
df=pd.DataFrame(allrec,columns=['strat','t','model','ui','j','p','lo','y','n','nm'])
df.to_pickle('calib.pkl')
def auc(p,y):
    r=rankdata(p); n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
out=[]
for (s,t),g in df.groupby(['strat','t']):
    p=g.p.values; y=g.y.values
    br=((p-y)**2).mean()
    ll=-(y*np.log(np.clip(p,1e-3,1))+(1-y)*np.log(np.clip(1-p,1e-3,1))).mean()
    # ECE 10 bins
    bins=np.minimum((p*10).astype(int),9); ece=sum(abs(p[bins==b].mean()-y[bins==b].mean())*(bins==b).mean() for b in range(10) if (bins==b).any())
    # brier model-bootstrap CI
    pm=g.assign(se=(g.p-g.y)**2).groupby('model').se.mean().values
    rg=np.random.default_rng(3); bs=[pm[rg.integers(0,len(pm),len(pm))].mean() for _ in range(1000)]
    out.append(dict(strat=s,t=t,n=len(g),base=y.mean(),brier=br,b_lo=np.percentile(bs,2.5),b_hi=np.percentile(bs,97.5),logloss=ll,auc=auc(p,y),ece=ece))
O=pd.DataFrame(out); O.to_csv('calib_summary.csv',index=False)
print(O.round(4).sort_values(['t','brier']).to_string())
