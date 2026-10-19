from common import *
rng=np.random.default_rng(9)
df=long_df(); L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
vpd=df.groupby(['model','ds']).vpd.first()
ims=[i for i in loss.index if i[1]!='sci']
def pref(m): return (('-hf' in m) or ('-merged' in m), m)
keep={}
for m in sorted({i[0] for i in ims},key=pref): keep.setdefault(twin(m),m)
ims=[i for i in ims if i[0] in set(keep.values())]
GEN3={'qwen3vl','qwen3.5'}
rows=[]
for (m,ds) in ims:
    rows.append(dict(model=m,ds=ds,gen3=int(GROUP[m] in GEN3) if m in GROUP else 0,lvpd=np.log10(vpd[(m,ds)]),hid=int(info(m)['dim']>128),f16=int(info(m)['f16']),info_ds=int(ds=='info'),
        **{l:loss.loc[(m,ds)][l] for l in L}))
D=pd.DataFrame(rows)
D['nonint4collapse']=~D.model.str.contains('nemotron')
print('n profiles',len(D),'models',D.model.nunique(),'gen3 models',D[D.gen3==1].model.nunique())
print(D.groupby('gen3')[['lvpd','hid','f16']].mean().round(2))
def ols(Dd,y,cols):
    X=np.column_stack([np.ones(len(Dd))]+[Dd[c].values.astype(float) for c in cols]); b,*_=np.linalg.lstsq(X,Dd[y].values,rcond=None); return b[1:]
def clus(y,cols,B=2000):
    b=ols(D,y,cols); ms=D.model.unique(); g={m:D[D.model==m] for m in ms}; bs=[]
    for _ in range(B):
        s=rng.choice(ms,len(ms)); d=pd.concat([g[m] for m in s]); 
        try: bs.append(ols(d,y,cols))
        except Exception: pass
    bs=np.array(bs); return b,np.percentile(bs,2.5,axis=0),np.percentile(bs,97.5,axis=0)
targets=['binary','hierarchical_merge(0.1)','lloyd2(7)','hierarchical_merge(0.25)','hierarchical_merge(0.33) > binary','adaptive_merge(0.6)']
print('--- coefficients on LOSS (1-retention); negative = lower loss. Units: points of retention x100')
for y in targets:
    for cols in (['gen3','info_ds'],['gen3','lvpd','hid','f16','info_ds']):
        b,lo,hi=clus(y,cols)
        print('%-36s'%y,' | '.join('%s %+.2f [%+.2f,%+.2f]'%(c,100*b[i],100*lo[i],100*hi[i]) for i,c in enumerate(cols) if c!='info_ds'))
# raw group means for the targets
M_=D.groupby(['gen3'])[targets].mean()*100; print((M_).round(2))
print((D.groupby(D.model.map(lambda m:GROUP2.get(m,'single')))[targets[:3]].mean()*100).round(2))
D.to_csv('s9_design.csv',index=False)
