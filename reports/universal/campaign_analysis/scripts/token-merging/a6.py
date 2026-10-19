import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(6)
W=pd.read_pickle('W.pkl')
R=[0.5,0.33,0.25,0.1]
rows=[]
for _,r in W.iterrows():
    for q in R:
        if not np.isnan(r[f'r{q}']): rows.append(dict(model=r.model,ds=r.ds,mod=r.modality,fam=r.family,q=q,abs=r[f'v{q}'],k=r[f'v{q}']/r.bv,loss=1-r[f'r{q}'],bv=r.bv,hi=r[f'hi{q}'],lo=r[f'lo{q}']))
    if not np.isnan(r.a_ret): rows.append(dict(model=r.model,ds=r.ds,mod=r.modality,fam=r.family,q='a',abs=r.a_v,k=r.a_v/r.bv,loss=1-r.a_ret,bv=r.bv,hi=r.a_hi,lo=r.a_lo))
L=pd.DataFrame(rows)
L.to_pickle('L.pkl')
H=L[L.q!='a'].copy()
print('--- loss by ABSOLUTE kept vectors per page (hierarchical cells), by dataset/modality')
bins=[0,20,30,50,75,100,150,250,500,2000]
H['absbin']=pd.cut(H['abs'],bins)
t=H.groupby(['absbin','ds']).loss.agg(['count','mean','median']).round(4).unstack('ds')
print(t.to_string())
print('\nmatched overlap abs in (50,150]: text vs image mean loss with model-bootstrap CI; and keep fractions involved')
for lo,hi in [(30,50),(50,100),(100,150)]:
    s=H[(H['abs']>lo)&(H['abs']<=hi)]
    tx=s[s['mod']=='text'];im=s[s['mod']=='image']
    if len(tx)<3 or len(im)<3: print(lo,hi,'n text',len(tx),'n image',len(im));continue
    bs=[rng.choice(tx.loss.values,len(tx)).mean()-rng.choice(im.loss.values,len(im)).mean() for _ in range(4000)]
    print((lo,hi),'text n=%d mean %.4f (median keep %.2f) | image n=%d mean %.4f (median keep %.2f) | diff %.4f CI %s'%(len(tx),tx.loss.mean(),tx.k.median(),len(im),im.loss.mean(),im.k.median(),tx.loss.mean()-im.loss.mean(),np.round(np.percentile(bs,[2.5,97.5]),4)))
print('\nFraction-matched: keep fraction in same bins of k (hier), text vs image mean loss')
for q in R:
    s=H[H.q==q];tx=s[s['mod']=='text'].loss;im=s[s['mod']=='image'].loss
    print(q,'text %.4f (abs median %.0f) image %.4f (abs median %.0f)'%(tx.mean(),s[s['mod']=='text']['abs'].median(),im.mean(),s[s['mod']=='image']['abs'].median()))
print('\n--- which predicts loss better: log(abs kept) or log(keep fraction)? Spearman with loss, pooled hier cells, by ds')
for ds,g in H.groupby('ds'):
    print(ds,'abs rho=%.2f  frac rho=%.2f  bv rho=%.2f'%(stats.spearmanr(g['abs'],g.loss)[0],stats.spearmanr(g.k,g.loss)[0],stats.spearmanr(g.bv,g.loss)[0]))
print('pooled all hier cells: abs rho=%.2f frac rho=%.2f'%(stats.spearmanr(H['abs'],H.loss)[0],stats.spearmanr(H.k,H.loss)[0]))
# OLS in logs: log(loss+0.002?) skip; use rank regression with partial: loss ~ ds dummies + log k + log bv
import numpy.linalg as la
Hh=H[H.loss>-1].copy()
X=np.column_stack([np.ones(len(Hh)),(Hh.ds=='infovqa').astype(float),(Hh.ds=='scifact').astype(float),np.log2(1/Hh.k)**2,np.log2(Hh.bv)])
y=Hh.loss.values;b=la.lstsq(X,y,rcond=None)[0]
groups={m:np.where(Hh.model.values==m)[0] for m in Hh.model.unique()};ms=list(groups);bs=[]
for _ in range(1000):
    idx=np.concatenate([groups[m] for m in rng.choice(ms,len(ms))]);bs.append(la.lstsq(X[idx],y[idx],rcond=None)[0])
bs=np.array(bs)
print('OLS loss ~ ds + (log2 1/k)^2 + log2(bv):',dict(zip(['int','infovqa','scifact','(log2 1/k)^2','log2 bv'],np.round(b,4))))
print(' model-bootstrap 95% CI for log2(bv) coef: ',np.round(np.percentile(bs[:,4],[2.5,97.5]),4),' for (log2 1/k)^2:',np.round(np.percentile(bs[:,3],[2.5,97.5]),4))
print('\n--- within-model docvqa vs infovqa: does larger bv in a dataset go with LESS loss? (Delta bv vs Delta loss)')
D=W[W.ds=='docvqa'].set_index('model');I=W[W.ds=='infovqa'].set_index('model');c=D.index.intersection(I.index)
for q in ['0.25','0.1']:
    dl=(1-D.loc[c,f'r{q}'])-(1-I.loc[c,f'r{q}']);db=np.log2(D.loc[c,'bv']/I.loc[c,'bv'])
    h=pd.concat([dl,db],axis=1).dropna();h.columns=['dl','db'];rho,p=stats.spearmanr(h.db,h.dl)
    nz=h[h.db.abs()>0.01]
    print(q,'all n=%d rho=%.2f p=%.2g | models whose vpd differs between datasets n=%d rho=%.2f p=%.2g; mean log2(bv_doc/bv_info)=%.2f'%(len(h),rho,p,len(nz),*stats.spearmanr(nz.db,nz.dl),db.mean()))
print('median bv docvqa %.0f infovqa %.0f; share of models with bv_doc>bv_info: %.2f'%(D.bv.median(),I.bv.median(),(D.loc[c,'bv']>I.loc[c,'bv']).mean()))
print('\n--- shape heterogeneity: cliff ratio L0.1/L0.25 and models with loss0.1 > 3x loss0.25 (+ >2pp)')
W['L25']=1-W['r0.25'];W['L10']=1-W['r0.1']
cl=W[(W.L10-W.L25>0.02)][['model','ds','L25','L10']];print(len(cl),'of',W.L10.notna().sum());print(cl.round(3).to_string())
print('\nfamilies n>=3 only eta^2 for L0.1 (image)')
im=W[W.modality=='image'].groupby(['family','model'])['L10'].mean().reset_index().dropna()
fam=im.groupby('family').model.transform('count');h=im[fam>=3]
gm=h.L10.mean();sst=((h.L10-gm)**2).sum();ssb=sum(len(g)*(g.L10.mean()-gm)**2 for _,g in h.groupby('family'));obs=ssb/sst
pm=[]
for _ in range(5000):
    p=h.copy();p['family']=rng.permutation(p.family.values);gm=p.L10.mean();pm.append(sum(len(g)*(g.L10.mean()-gm)**2 for _,g in p.groupby('family'))/sst)
print('families',h.family.unique(),'n=%d eta2=%.2f perm p=%.3g'%(len(h),obs,(np.sum(np.array(pm)>=obs)+1)/5001))
