import sys, pickle, itertools
from common import *
rng=np.random.default_rng(0)
U=load()
cnt=collections.Counter(k for u in U for k in u['rows'])
CAND=[k for k,n in cnt.items() if n>=86 and k!=(None,None,'f32')]
CAND.sort(key=key_str)
nC=len(CAND); nU=len(U)
R=np.full((nU,nC),np.nan); L=R.copy(); C=R.copy()
for i,u in enumerate(U):
    for j,k in enumerate(CAND):
        r=u['rows'].get(k)
        if r: R[i,j]=r['retention']; L[i,j]=r['retention_interval'][0]; C[i,j]=r['compression']
models=sorted({u['model'] for u in U})
midx={m:[i for i,u in enumerate(U) if u['model']==m] for m in models}
mod=np.array([u['modality'] for u in U]); ds=np.array([u['ds'] for u in U])
fam=np.array([u['family'] for u in U]); clu=np.array([u['cluster'] for u in U])
mname=np.array([u['model'] for u in U])
minfo={m:U[midx[m][0]] for m in models}
TARGETS=[0.95,0.97,0.99]
f16=CAND.index((None,None,'f16')); i8=CAND.index((None,None,'int8'))

def feat(m, size=False):
    u=minfo[m]; v=[math.log2(u['dim']), math.log2(u['vpd']), 8.0*(u['modality']=='image')]
    if size: v.append(math.log2(u['size']))
    return np.array(v)
F={s:{m:feat(m,s) for m in models} for s in (False,True)}
def knn_models(m, pool, k, size, mod_hard):
    pool=[p for p in pool if p!=m]
    allf=np.array([F[size][p] for p in models]); sd=allf.std(0); sd[sd==0]=1
    # modality feature already weighted by 8 (so it dominates) -> rescale: standardise others only
    def z(p):
        v=F[size][p].copy()
        return v
    cols=[0,1]+([3] if size else [])
    base=F[size][m]
    d=[]
    for p in pool:
        dd=sum(((F[size][p][c]-base[c])/ (sd[c]))**2 for c in cols)
        if mod_hard: dd+= 100*(minfo[p]['modality']!=minfo[m]['modality'])
        d.append(dd)
    o=np.argsort(d)[:k]; return [pool[i] for i in o]

def select(rows, t, rule):
    """rows: unit indices. returns candidate index or None."""
    r=R[rows]; l=L[rows]; c=C[rows]
    avail=~np.isnan(r)
    n=avail.sum(0)
    if rule=='min': ok=np.nanmin(np.where(avail,r,np.inf),0)>=t
    elif rule=='min+.01': ok=np.nanmin(np.where(avail,r,np.inf),0)>=t+0.01
    elif rule=='q90': ok=(np.where(avail,r>=t,False).sum(0)/np.maximum(n,1))>=0.9
    elif rule=='lbmin': ok=np.nanmin(np.where(avail,l,np.inf),0)>=t
    ok&=n>=max(1,int(0.8*len(rows)))
    if not ok.any(): return None
    med=np.nanmedian(np.where(avail,c,np.nan),0)
    mn=np.nanmin(np.where(avail,r,np.inf),0)
    score=np.where(ok,med+1e-6*mn,-np.inf)
    return int(np.argmax(score))

def oracle(i,t,crit):
    x=(R if crit=='pt' else L)[i]>=t
    if not np.isfinite(C[i][x]).any() or not x.any(): return None
    return int(np.nanargmax(np.where(x,C[i],-np.inf)))

def trainset(m, strat, clus_out):
    others=[p for p in models if p!=m]
    if clus_out: others=[p for p in others if minfo[p]['cluster']!=minfo[m]['cluster']]
    um=minfo[m]
    if strat=='global': ms=others
    elif strat=='modality': ms=[p for p in others if minfo[p]['modality']==um['modality']]
    elif strat=='family':
        ms=[p for p in others if minfo[p]['family']==um['family']]
        if len(ms)<2: ms=[p for p in others if minfo[p]['modality']==um['modality']]  # fallback flagged
    elif strat=='famout':
        ms=[p for p in others if minfo[p]['family']!=um['family']]
    elif strat.startswith('knn'):
        k=int(strat[3:].split('_')[0]); size='size' in strat; hard='nomod' not in strat
        ms=knn_models(m,others,k,size,hard)
    elif strat=='dsonly': ms=others
    return ms

def run():
    out=[]
    STR=['global','modality','family','famout','knn5','knn10','knn5_nomod','knn5_size','dsonly']
    for strat in STR:
      for co in (False,True):
        for m in models:
            ms=trainset(m,strat,co)
            for ui in midx[m]:
                if strat=='dsonly': rows=[i for p in ms for i in midx[p] if ds[i]==ds[ui]]
                else: rows=[i for p in ms for i in midx[p]]
                if strat=='famout' and co: pass
                for t in TARGETS:
                    for rule in ('min','min+.01','q90','lbmin'):
                        j=select(rows,t,rule) if rows else None
                        fb=False
                        if j is None: j=f16; fb=True
                        ret=R[ui,j]; lo=L[ui,j]; comp=C[ui,j]
                        o_pt=oracle(ui,t,'pt'); o_lb=oracle(ui,t,'lb')
                        out.append(dict(strat=strat,co=co,model=m,ui=ui,ds=ds[ui],mod=mod[ui],fam=fam[ui],t=t,rule=rule,
                            j=j,fb=fb,ret=ret,lo=lo,comp=comp,
                            comp_o_pt=C[ui,o_pt] if o_pt is not None else np.nan,
                            comp_o_lb=C[ui,o_lb] if o_lb is not None else np.nan,
                            ntrain=len(ms)))
    return out
if __name__=='__main__':
    out=run()
    import pandas as pd
    df=pd.DataFrame(out); df.to_pickle('lomo.pkl')
    print(len(df), nC, 'candidates')
    print([key_str(k) for k in CAND])
