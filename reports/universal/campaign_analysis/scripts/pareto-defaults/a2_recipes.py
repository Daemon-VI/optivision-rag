from common import *
m=get(); keys=sorted(m.key.unique())
info=m.groupby('key').agg(modal=('modal','first'),ds=('ds','first'),model=('model','first'),family=('family','first'),dim=('dim','first'),base=('base','first'),vpd=('vpd','max'),legacy=('legacy','first'))
print("pairs",len(keys),info.modal.value_counts().to_dict(),info.ds.value_counts().to_dict(),"legacy",info.legacy.sum())
rows=[]
for lab,g in m.groupby('label'):
    n=g.key.nunique()
    if n<60: continue
    rows.append(dict(label=lab,n=n,comp_med=g.comp.median(),comp_min=g.comp.min(),stages=g.stages.iloc[0]))
L=pd.DataFrame(rows).set_index('label')
def cov(g,T,col='lo',universe=None): return int((g[col]>=T).sum())
out=[]
for T in (.95,.97,.99):
  for col in ('lo','ret'):
    for sub in ('all','image','text'):
        for lab in L.index:
            g=m[m.label==lab]
            if sub!='all': g=g[g.modal==sub]
            k=int((g[col]>=T).sum()); n=len(g); lo_,hi_=wilson(k,n)
            out.append(dict(thr=T,col=col,sub=sub,label=lab,k=k,n=n,cov=k/n,wlo=lo_,whi=hi_,comp_med=L.loc[lab,'comp_med'],comp_min=L.loc[lab,'comp_min']))
O=pd.DataFrame(out); O.to_csv('recipes_all.csv',index=False)
print("\nRECIPE TABLE (col=lo means 95% bootstrap CI lower bound >= T). Best median compression with coverage>=C among labels measured on >=60 pairs; denominators = pairs where the label exists")
for sub in ('all','image','text'):
  for T in (.95,.97,.99):
    for C in (.9,.95,1.0):
        q=O[(O.thr==T)&(O.col=='lo')&(O['sub']==sub)&(O['cov']>=C-1e-9)].sort_values('comp_med',ascending=False)
        if len(q)==0:
            q2=O[(O.thr==T)&(O.col=='lo')&(O['sub']==sub)].sort_values(['cov','comp_med'],ascending=False).iloc[0]
            print(f"{sub:5s} T={T} C={C}: none; best cov {q2['cov']:.3f} ({q2.label})"); continue
        x=q.iloc[0]; print(f"{sub:5s} T={T} C={C}: {x.label} {x.k}/{x.n} cov={x['cov']:.3f} Wilson[{x.wlo:.2f},{x.whi:.2f}] comp_med={x.comp_med:.1f}x min={x.comp_min:.1f}x")
