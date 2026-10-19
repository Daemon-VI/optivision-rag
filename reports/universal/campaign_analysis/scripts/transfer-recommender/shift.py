from lomo import *
import pandas as pd
from scipy.stats import beta
cmed=np.nanmedian(C,0)
band=lambda d: 0 if d<=96 else 1 if d==128 else 2 if d<=640 else 3
dsA='vidore/docvqa_test_subsampled'; dsB='vidore/infovqa_test_subsampled'
rg=np.random.default_rng(8)
PRI=4.0
rows=[]
for t in TARGETS:
  for m in models:
    um=minfo[m]
    if um['modality']!='image': continue
    oth=[p for p in models if p!=m and minfo[p]['cluster']!=um['cluster'] and minfo[p]['modality']=='image' and band(minfo[p]['dim'])==band(um['dim'])]
    if not oth: continue
    allr=[i for p in models if p!=m and minfo[p]['cluster']!=um['cluster'] and minfo[p]['modality']=='image' for i in midx[p]]
    gh=np.array([(R[allr,j]>=t).sum() for j in range(nC)]); gn=np.array([(~np.isnan(R[allr,j])).sum() for j in range(nC)]); gp=(gh+.5)/(gn+1)
    for ui in midx[m]:
        for mode in ('same','other','pooled'):
            tr=[i for p in oth for i in midx[p] if (mode=='pooled' or (ds[i]==ds[ui])==(mode=='same'))]
            if not tr: continue
            r=R[tr]; av=~np.isnan(r); h=((r>=t)&av).sum(0); n=av.sum(0)
            a=PRI*gp; b=PRI*(1-gp)
            p=(h+a)/(n+PRI); lo=beta.ppf(0.1,h+a,n-h+b)
            ok=np.where(lo>=0.9)[0]
            j=int(ok[np.argmax(cmed[ok])]) if len(ok) else f16
            ya=(R[ui]>=t); okc=~np.isnan(R[ui])
            o=oracle(ui,t,'pt')
            rows.append(dict(t=t,model=m,ui=ui,ds=ds[ui][-22:-14],band=band(um['dim']),mode=mode,hit=bool(R[ui,j]>=t),comp=C[ui,j],ocomp=C[ui,o],short=t-R[ui,j],
               brier=np.mean((p[okc]-ya[okc])**2),pipe=key_str(CAND[j])))
D=pd.DataFrame(rows)
D.to_pickle('shift.pkl')
def cb(g):
    per=g.groupby('model').hit.mean().values
    bs=[per[rg.integers(0,len(per),len(per))].mean() for _ in range(1000)]
    return pd.Series(dict(n=len(g),hit=g.hit.mean(),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),med_comp=g.comp.median(),brier=g.brier.mean()))
pd.set_option('display.width',200)
print(D.groupby(['t','ds','mode']).apply(cb).round(3))
print(D[D.band==1].groupby(['t','ds','mode']).apply(cb).round(3))
