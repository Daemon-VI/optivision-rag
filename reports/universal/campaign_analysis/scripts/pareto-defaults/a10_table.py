from common import *
m=get()
rec=['float16','int8(per_vector)','int4(mean)','hierarchical_merge(0.5)','hierarchical_merge(0.33) > int8(per_vector)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.33) > int4(mean)','hierarchical_merge(0.25) > int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.25) > lloyd2(7)','hierarchical_merge(0.25) > binary']
subs={'ALL':lambda g:g,'image':lambda g:g[g.modal=='image'],'docvqa':lambda g:g[g.ds=='docvqa'],'infovqa':lambda g:g[g.ds=='infovqa'],'text(scifact)':lambda g:g[g.modal=='text'],'dim>128':lambda g:g[g.dim>128]}
rows=[]
for l in rec:
    g0=m[m.label==l]
    for sn,f in subs.items():
        g=f(g0)
        for T in (.95,.97,.99):
            k=int((g.lo>=T).sum()); n=len(g); w=wilson(k,n)
            rows.append(dict(label=l,sub=sn,thr=T,k=k,n=n,cov=k/n,wlo=w[0],whi=w[1],comp_med=g0.comp.median(),comp_min=g0.comp.min(),ret_med=g.ret.median(),ret_min=g.ret.min()))
R=pd.DataFrame(rows); R.to_csv('final_recipe_table.csv',index=False)
for T in (.95,.97,.99):
    print(f"\n### lower bound >= {T}: k/n (Wilson 95%) ; ALL = every measured pair for that label (95 for codec-only labels, 88 otherwise)")
    p=R[R.thr==T].copy(); p['s']=p.apply(lambda r:f"{r.k}/{r.n} [{r.wlo:.2f}-{r.whi:.2f}]",axis=1)
    t=p.pivot(index='label',columns='sub',values='s')[list(subs)]; t['comp_med']=p.groupby('label').comp_med.first().round(1); t['comp_min']=p.groupby('label').comp_min.first().round(1)
    print(t.loc[rec].to_string())
# int8 failures at .99
for l in ['int8(per_vector)','hierarchical_merge(0.5)','int4(mean)']:
    g=m[(m.label==l)&(m.lo<.99)].sort_values('lo'); print('\n',l,'lo<.99:',len(g)); 
    if l!='int4(mean)' : print(g[['key','lo','ret']].head(8).round(3).to_string(index=False))
# coverage-compression frontier over all labels (n>=88) per subgroup/T: non-dominated by (comp_med, cov)
print("\n== coverage frontier (labels non-dominated in median compression and coverage), image-only and text-only ==")
for sn in ('image','text(scifact)'):
  for T in (.95,.99):
    out=[]
    for l,g0 in m.groupby('label'):
        if g0.key.nunique()<60: continue
        g=subs[sn](g0)
        if len(g)<10: continue
        out.append((l,g0.comp.median(),(g.lo>=T).mean(),len(g),(g.lo>=T).sum()))
    D=pd.DataFrame(out,columns=['label','comp','cov','n','k']).sort_values(['comp','cov'],ascending=[False,False])
    best=-1;keep=[]
    for _,r in D.iterrows():
        if r['cov']>best+1e-12: keep.append(r); best=r['cov']
    print(f"\n{sn} T={T}:"); print(pd.DataFrame(keep).sort_values('comp').round(2).to_string(index=False))
