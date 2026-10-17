"""Exploratory predictor tests for plain-int4 degradation. Model-level (mean over datasets). Counts every test in NT."""
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(0)
NT=0
def T(name,*vals):
    global NT; NT+=1; print(f'[T{NT}] {name}:',*vals)
df=pd.read_pickle('master2.pkl')
g=df.groupby('model').agg(mod=('modality','first'),fam=('family','first'),dim=('dim','first'),dt=('dtype','first'),vpd=('vpd','mean'),
    base=('base5','mean'),int4=('int4','mean'),int4m=('int4m','mean'),bin=('bin','mean'),int8=('int8','mean'),l2=('l2','mean'),dl=('downloads','first')).reset_index()
g['deg']=g.int4<0.95; g['col']=g.int4<0.5
print('models',len(g),'degraded<0.95:',g.deg.sum(),'collapsed<0.5:',g.col.sum())
print(g.groupby('mod')[['deg','col']].agg(['sum','count']))
# T1 modality
for lab in ['deg','col']:
    tab=pd.crosstab(g['mod'],g[lab]).reindex(columns=[False,True],fill_value=0)
    o,p=stats.fisher_exact(tab.values); T(f'Fisher modality x {lab}',tab.values.tolist(),'OR',round(o,2) if np.isfinite(o) else o,'p',round(p,4))
# dataset-confound-free comparison impossible: modality==dataset
# T: continuous predictors, Spearman vs int4 (all models) and within modality
def spear(x,y,B=2000):
    r=stats.spearmanr(x,y); n=len(x); bs=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(set(x.iloc[i]))<2 or len(set(y.iloc[i]))<2: continue
        bs.append(stats.spearmanr(x.iloc[i],y.iloc[i])[0])
    return r[0],r[1],np.nanpercentile(bs,[2.5,97.5])
g['ldim']=np.log2(g.dim)
for sub,gg in [('all',g),('image',g[g['mod']=='image']),('text',g[g['mod']=='text'])]:
    for p in ['ldim','vpd','base']:
        r,pv,ci=spear(gg[p],gg.int4)
        T(f'Spearman int4 ~ {p} [{sub}, n={len(gg)}]','rho',round(r,2),'p',round(pv,3),'boot95',np.round(ci,2))
# dtype among image models (known dtype) ; restrict dim 128 to avoid dimension confound
im=g[(g['mod']=='image')&(g.dt!='unknown/auto')]
for lab,gg in [('image all',im),('image 128-d',im[im.dim==128])]:
    a=gg[gg.dt=='float32'].int4; b=gg[gg.dt=='float16-inference'].int4
    u=stats.mannwhitneyu(a,b)
    T(f'MWU int4 float32 vs float16-inference [{lab}]',f'n={len(a)},{len(b)}',f'median {a.median():.3f} vs {b.median():.3f}','p',round(u.pvalue,3),'; collapsed',int((a<.5).sum()),int((b<.5).sum()))
# high-dim models
print(g[g.dim>=1024][['model','dim','dt','int4','int4m','bin','int8']].round(3).to_string())
print(g[(g.dim>=320)&(g.dim<1024)][['model','dim','dt','int4']].round(3).to_string())
# dim classes
g['dcls']=pd.cut(g.dim,[0,64,128,640,5000],labels=['<=64','96-128','320-640','>=2048'])
print(g.groupby(['mod','dcls'],observed=True).int4.agg(['count','min','median','max']).round(3))
# T: signature relationships among models: int4 vs binary / int8 / lloyd2
for sub,gg in [('all',g),('text',g[g['mod']=='text']),('image',g[g['mod']=='image'])]:
    for c in ['bin','int8','l2']:
        h=gg.dropna(subset=[c]); r,pv,ci=spear(h.int4,h[c])
        T(f'Spearman int4 ~ {c} [{sub}, n={len(h)}]','rho',round(r,2),'p',round(pv,4),'boot95',np.round(ci,2))
# Row-level int8 vs int4 : slope on deficit
d=df.dropna(subset=['int4','int8'])
r=stats.spearmanr(d.int4,d.int8); T('Spearman rows int4~int8 n=',len(d),round(r[0],2),round(r[1],4))
# int8 deficit among collapsed vs rest
col=d[d.int4<0.5]; rest=d[d.int4>=0.9]
print('int8 mean collapsed',col.int8.mean().round(4),len(col),'rest>=0.9',rest.int8.mean().round(4),len(rest))
u=stats.mannwhitneyu(col.int8,rest.int8,alternative='less'); T('MWU int8 collapsed < int8 rest','p',round(u.pvalue,4))
print(col[['model','dataset','int4','int8','int8(per_vector)|lo','int8(per_vector)|hi']].round(3).to_string())
# int4(mean) residuals
dm=df.dropna(subset=['int4m'])
print('int4m n',len(dm),'min',dm.int4m.min().round(3),'rows<0.98:',(dm.int4m<0.98).sum(),dm[dm.int4m<0.98][['model','dataset','int4m','int4']].round(3).to_string())
# signature types among degraded models
dg=g[g.deg].copy(); dg['bin_ok']=dg.bin>0.95
print(dg[['model','mod','dim','int4','bin','int8','l2','int4m','bin_ok']].round(3).to_string())
# hub weighting
tot=g.dl.sum(); print('downloads total (measured set)',tot,'degraded share',g[g.deg].dl.sum()/tot, 'collapsed share',g[g.col].dl.sum()/tot)
tx=g[g['mod']=='text']; print('text only: downloads',tx.dl.sum(),'degraded share',tx[tx.deg].dl.sum()/tx.dl.sum())
r=stats.spearmanr(np.log10(tx.dl+1),tx.int4); T('Spearman text int4 ~ log10 downloads',round(r[0],2),round(r[1],3))
print('N TESTS',NT)
g.to_pickle('model_level.pkl')
