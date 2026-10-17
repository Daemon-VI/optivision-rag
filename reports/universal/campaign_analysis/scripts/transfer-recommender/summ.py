import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',500)
df=pd.read_pickle('lomo.pkl')
df=df[df.ret.notna()].copy()
df['hit_pt']=df.ret>=df.t; df['hit_lb']=df.lo>=df.t
df['ratio_pt']=df.comp/df.comp_o_pt
df['short']=df.t-df.ret
rng=np.random.default_rng(1)
def agg(g):
    # cluster bootstrap by model
    ms=g.model.unique(); per=g.groupby('model').agg(h=('hit_pt','mean'),hl=('hit_lb','mean'),n=('hit_pt','size'))
    hp=per.h.mean()
    bs=[]
    for _ in range(1000):
        s=rng.choice(len(per),len(per)); bs.append(per.h.values[s].mean())
    nf=g.fb.mean()
    h=g[g.hit_pt]
    return pd.Series(dict(n_units=len(g),hit_pt=g.hit_pt.mean(),hit_pt_ci_lo=np.percentile(bs,2.5),hit_pt_ci_hi=np.percentile(bs,97.5),hit_lb=g.hit_lb.mean(),
        fallback=nf,med_comp=g.comp.median(),med_ratio_hits=h.ratio_pt.median(),mean_shortfall_fail=g.short[~g.hit_pt].mean(),
        oracle_med=g.comp_o_pt.median()))
res=df.groupby(['t','rule','strat','co']).apply(agg).reset_index()
res.to_csv('lomo_summary.csv',index=False)
for t in (0.95,0.97,0.99):
  print('=== target',t)
  print(res[(res.t==t)&(res.co==True)].round(3).drop(columns=['t','co']).to_string())
