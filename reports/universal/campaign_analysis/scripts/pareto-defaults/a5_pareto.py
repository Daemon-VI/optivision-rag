from common import *
m=get()
G=m[~m.legacy].copy()   # 88 pairs on the common grid
print("grid pairs",G.key.nunique(), G.groupby('modal').key.nunique().to_dict())
def front(df,ycol):
    d=df.sort_values(['comp',ycol],ascending=[False,False])
    best=-9; keep=[]
    for i,r in d.iterrows():
        if r[ycol]>best+1e-12: keep.append(i); best=r[ycol]
    return keep
# 1) front membership on point retention and on lower bound, restricted to retention>=0.9 (the useful region)
cnt={}
for k,g in G.groupby('key'):
    for ycol in ('ret','lo'):
        for i in front(g,ycol):
            lab=G.loc[i,'label']; cnt.setdefault((ycol,lab),set()).add(k)
FR=pd.DataFrame([dict(ycol=a,label=b,n=len(v)) for (a,b),v in cnt.items()])
print("\nmean front size per pair:", {y:np.mean([len(front(g,y)) for _,g in G.groupby('key')]) for y in ('ret','lo')})
for y in ('ret','lo'):
    f=FR[FR.ycol==y].sort_values('n',ascending=False)
    print(f"\nfront on {y}: labels by # pairs on front (of 88)"); print(f.head(15).to_string(index=False))
# 2) oracle: max-compression label with lo>=T per pair
oracle=[]
for k,g in G.groupby('key'):
    for T in (.95,.97,.99):
        q=g[g.lo>=T]; b=q.sort_values('comp',ascending=False).iloc[0]
        oracle.append(dict(key=k,thr=T,label=b.label,comp=b.comp,modal=b.modal,ds=b.ds,dim=b.dim,nsel=len(g)))
O=pd.DataFrame(oracle); O.to_csv('oracle.csv',index=False)
for T in (.95,.97,.99):
    o=O[O.thr==T]
    print(f"\nORACLE T={T}: per-pair max compression among {int(o.nsel.iloc[0])}~ labels: median {o.comp.median():.1f}x, IQR {o.comp.quantile(.25):.1f}-{o.comp.quantile(.75):.1f}, by modal:",o.groupby('modal').comp.median().round(1).to_dict(), "by ds:",o.groupby('ds').comp.median().round(1).to_dict())
    print(o.label.value_counts().head(8).to_string())
# oracle selection bias: the oracle is a max over ~54-88 labels on a CI lower bound -> check its held-out? (see selection_rules). Report only descriptive.
# 3) LOMO: choose label maximizing median compression s.t. coverage over other models' pairs >= C (lo>=T), test on held-out model's pairs
labs=[l for l,g in G.groupby('label') if g.key.nunique()==88]
P=G[G.label.isin(labs)].pivot_table(index='key',columns='label',values='lo')
Cm=G[G.label.isin(labs)].groupby('label').comp.median()
model_of={k:k.split('|')[0] for k in P.index}
fam_of={k:G[G.key==k].family.iloc[0] for k in P.index}
res=[]
for T in (.95,.97,.99):
  for C in (.90,.95,1.0):
    for unit,mp in (('model',model_of),('family',fam_of)):
        hit=0;tot=0;chosen={};comps=[]
        for u in sorted(set(mp.values())):
            tr=[k for k in P.index if mp[k]!=u]; te=[k for k in P.index if mp[k]==u]
            cov=(P.loc[tr]>=T).mean()
            ok=cov[cov>=C-1e-9]
            if len(ok)==0: lab='float32'
            else: lab=Cm[ok.index].sort_values(ascending=False).index[0]
            chosen[lab]=chosen.get(lab,0)+1
            hit+=int((P.loc[te,lab]>=T).sum()); tot+=len(te); comps.append(Cm[lab])
        res.append(dict(thr=T,C=C,unit=unit,heldout_cov=hit/tot,k=hit,n=tot,median_comp=np.median(comps),chosen=chosen))
R=pd.DataFrame(res); pd.set_option('display.width',250,'display.max_colwidth',120); print("\nLEAVE-ONE-UNIT-OUT default selection (88 grid pairs)"); print(R.to_string(index=False))
R.to_csv('lomo.csv',index=False)
