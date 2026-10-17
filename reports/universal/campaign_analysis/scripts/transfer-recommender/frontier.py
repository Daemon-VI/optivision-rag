from lomo import *
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
df=pd.read_pickle('calib.pkl')
cmed=np.nanmedian(C,0)
rg=np.random.default_rng(11)
res=[]
detail=[]
for (s,t),g in df.groupby(['strat','t']):
    if s not in ('global','modality','dimband_mod','family','knn10','dknn5'): continue
    for qn in ('p','lo'):
      for c in (0.5,0.7,0.8,0.9,0.95,0.98):
        for (m,ui),gg in g.groupby(['model','ui']):
            ok=gg[gg[qn]>=c]
            if len(ok)==0: j=f16; fb=True
            else: j=int(ok.j.values[np.argmax(cmed[ok.j.values])]); fb=False
            o=oracle(ui,t,'pt')
            detail.append((s,t,qn,c,m,ui,j,fb,R[ui,j]>=t,L[ui,j]>=t,C[ui,j],C[ui,o] if o is not None else np.nan, t-R[ui,j]))
D=pd.DataFrame(detail,columns=['strat','t','q','c','model','ui','j','fb','hit','hitlb','comp','ocomp','short'])
D.to_pickle('frontier.pkl')
D['ratio']=D.comp/D.ocomp
def cb(g):
    per=g.groupby('model').hit.mean().values
    bs=[per[rg.integers(0,len(per),len(per))].mean() for _ in range(500)]
    return pd.Series(dict(n=len(g),hit=g.hit.mean(),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),hitlb=g.hitlb.mean(),fb=g.fb.mean(),
       med_comp=g.comp.median(),mean_log2_ratio=np.log2(g.ratio).mean(),eff_comp_med=np.where(g.hit,g.comp,2.0).mean(), med_short_fail=g.short[~g.hit].median() if (~g.hit).any() else np.nan,
       max_short=g.short[~g.hit].max() if (~g.hit).any() else np.nan))
F=D.groupby(['t','strat','q','c']).apply(cb).reset_index()
F.to_csv('frontier_summary.csv',index=False)
for t in TARGETS:
    print('=== t',t)
    print(F[(F.t==t)&(F.q=='lo')&(F.strat.isin(['global','dimband_mod','family','knn10']))].round(3).drop(columns=['t','q']).to_string())
