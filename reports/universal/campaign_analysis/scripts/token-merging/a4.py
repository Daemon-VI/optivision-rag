import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(3)
W=pd.read_pickle('W.pkl')
R=[0.5,0.33,0.25,0.1]
for r in R: W[f'L{r}']=1-W[f'r{r}']; W[f'se{r}']=(W[f'hi{r}']-W[f'lo{r}'])/3.92
W['La']=1-W.a_ret;W['sea']=(W.a_hi-W.a_lo)/3.92
print('--- noise floor: mean query-bootstrap SE of retention vs between-model SD of retention')
for ds,g in W.groupby('ds'):
    for r in R+['a']:
        L=g['La' if r=='a' else f'L{r}'].dropna();se=g['sea' if r=='a' else f'se{r}'].loc[L.index]
        print(ds,r,'SD(loss)=%.4f  rms SE=%.4f  share of variance that is sampling noise ~%.0f%%  mean CI half-width %.4f'%(L.std(),np.sqrt((se**2).mean()),100*(se**2).mean()/L.var(),1.96*se.mean()))
print('\nretention>1 share', {ds:round((g[[f'r{r}' for r in R]].stack()>1).mean(),3) for ds,g in W.groupby('ds')})
print('\n--- cells whose CI upper bound < 0.99 / < 0.95 (credibly lossy), by ratio and ds')
for ds,g in W.groupby('ds'):
    for r in R:
        h=g[f'hi{r}'].dropna();print(ds,r,'n=%d hi<0.99: %d  hi<0.95: %d  lo>=0.99: %d'%(len(h),(h<0.99).sum(),(h<0.95).sum(),(g[f'lo{r}'].dropna()>=0.99).sum()))
print('\n--- Adaptive(0.6) vs hierarchical at matched keep fraction (hier interpolated log-linearly in log keep, loss scale)')
res=[]
for _,r in W.iterrows():
    if np.isnan(r.a_ret): continue
    ks=[];ls=[]
    for q in [0.5,0.33,0.25,0.1]:
        if not np.isnan(r[f'r{q}']): ks.append(r[f'v{q}']/r.bv);ls.append(1-r[f'r{q}'])
    ks=[1]+ks;ls=[0]+ls
    ka=r.a_v/r.bv
    if ka<min(ks) or ka>1: continue  # outside measured range
    lh=np.interp(np.log(ka),np.log(ks[::-1]),ls[::-1])
    res.append(dict(model=r.model,ds=r.ds,mod=r.modality,ka=ka,la=1-r.a_ret,lh=lh,d=(1-r.a_ret)-lh))
res=pd.DataFrame(res)
print('cells in range',len(res),'of',W.a_ret.notna().sum(), 'excluded (keep<0.1):',W.a_ret.notna().sum()-len(res))
for ds,g in res.groupby('ds'):
    d=g.d.values;bs=np.percentile([rng.choice(d,len(d)).mean() for _ in range(5000)],[2.5,97.5])
    print(ds,'n=%d median keep=%.3f; adaptive loss %.4f vs hier-interp loss %.4f; mean diff (adaptive-hier)=%.4f CI %s; adaptive worse in %d; wilcoxon p=%.2g'%(len(g),g.ka.median(),g.la.mean(),g.lh.mean(),d.mean(),np.round(bs,4),(d>0).sum(),stats.wilcoxon(d).pvalue))
print(res.sort_values('d').tail(5).round(3).to_string())
print('\nadaptive keep fraction (radius .6) across models: range per ds')
W['ak']=W.a_v/W.bv
print(W.groupby('ds').ak.describe().round(3))
print('adaptive keep vs loss spearman (within ds)')
for ds,g in W.groupby('ds'):
    print(ds,'rho=%.2f p=%.2g'%stats.spearmanr(g.ak,g.La,nan_policy='omit')[:2])
print('\nadaptive radius 0.5/0.7 vs 0.6 for models that have them')
import json
from load import load
df,_=load()
ad=df[df.stage=='adaptive_merge']
print(ad.groupby(['model','ds','radius']).agg(ret=('ret','mean'),vpd=('vpd','mean')).round(3).to_string() if ad.radius.nunique()>1 else '')
print('\nWard max_distance rows')
wd=df[(df.stage=='hierarchical_merge')&df.ratio.isna()]
print(wd.groupby(['model','ds']).size())
x=wd.copy();x['d']=x.label.str.extract(r'd=([0-9.]+)').astype(float)
print(x[['model','ds','d','vpd','ret','comp']].round(3).to_string())
print('\nAdaptive failures (retention<0.9 or vpd<5):')
print(W[(W.a_ret<0.93)|(W.a_v<5)][['model','ds','a_v','a_ret']].round(3))
