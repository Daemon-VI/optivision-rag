import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(1)
W=pd.read_pickle('W.pkl')
R=[0.5,0.33,0.25,0.1]
for r in R: W[f'L{r}']=1-W[f'r{r}']; W[f'abs{r}']=W[f'v{r}']
W['La']=1-W.a_ret
print('--- functional form of loss vs keep fraction k (pooled per dataset), through origin at k=1')
for ds,g in W.groupby('ds'):
    recs=[]
    for r in R:
        h=g[['model',f'v{r}',f'L{r}','bv']].dropna()
        recs.append(pd.DataFrame(dict(model=h.model,k=h[f'v{r}']/h.bv,L=h[f'L{r}'])))
    Lg=pd.concat(recs)
    X=Lg.k.values;Y=Lg.L.values
    for name,z in [('log2(1/k)',np.log2(1/X)),('1-k',1-X),('1/k-1',1/X-1),('(log2 1/k)^2',np.log2(1/X)**2)]:
        b=(z@Y)/(z@z);print(ds,name,'slope=%.4f SSE=%.4f'%(b,((Y-b*z)**2).sum()))
    def fitp(X,Y,step):
        bb=None
        for p in np.arange(0.5,4,step):
            z=np.log2(1/X)**p;b=(z@Y)/(z@z);sse=((Y-b*z)**2).sum()
            if bb is None or sse<bb[0]:bb=(sse,p,b)
        return bb
    best=fitp(X,Y,0.05);ps=[]
    groups={m:d for m,d in Lg.groupby('model')};ms=list(groups)
    for _ in range(300):
        s_=rng.choice(ms,len(ms));gg=pd.concat([groups[m] for m in s_]);ps.append(fitp(gg.k.values,gg.L.values,0.1)[1])
    print(ds,'power p=%.2f s=%.4f ; model-bootstrap 95%% p: %s'%(best[1],best[2],np.round(np.percentile(ps,[2.5,97.5]),2)))
print('\n--- redundancy: Spearman(base vectors/page, loss) by ds and ratio; also abs kept vectors')
def sp(x,y):
    m=~(np.isnan(x)|np.isnan(y));return stats.spearmanr(x[m],y[m]),m.sum()
nt=0
for ds,g in W.groupby('ds'):
    for r in R+['a']:
        L=g['La' if r=='a' else f'L{r}'].values; bvv=g.bv.values
        (rho,p),n=sp(bvv,L);nt+=1
        # cluster bootstrap by family
        fams=g.family.unique();bs=[]
        for _ in range(500):
            s=rng.choice(fams,len(fams));gg=pd.concat([g[g.family==f] for f in s])
            (rr,_),_n=sp(gg.bv.values,gg['La' if r=='a' else f'L{r}'].values)
            bs.append(rr)
        bs=np.array(bs);bs=bs[~np.isnan(bs)]
        print(ds,r,'n=%d rho=%.2f p=%.3g family-bootstrap 95%% %s'%(n,rho,p,np.round(np.percentile(bs,[2.5,97.5]),2)))
print('\nWithin-image: dim, baseline ndcg, bv vs loss at 0.1 (Spearman) per ds')
for ds,g in W[W.modality=='image'].groupby('ds'):
    for c in ['bv','dim','base']:
        (rho,p),n=sp(g[c].values.astype(float),g['L0.1'].values);print(ds,c,'rho=%.2f p=%.3g n=%d'%(rho,p,n))
print('\nGroup bv strata (image), mean loss at 0.1 and 0.25')
im=W[W.modality=='image'].copy()
im['bvbin']=pd.cut(im.bv,[0,500,800,1100,2000])
print(im.groupby(['ds','bvbin'])[['L0.25','L0.1','La']].agg(['mean','count']).round(4))
print('\nAbsolute kept vectors at 0.1 vs loss, image: ')
for ds,g in im.groupby('ds'):
    (rho,p),n=sp(g['abs0.1'].values,g['L0.1'].values);print(ds,'abs0.1 rho=%.2f p=%.3g'%(rho,p))
print('\nText: bv vs loss')
t=W[W.modality=='text']
for r in R:
    (rho,p),n=sp(t.bv.values,t[f'L{r}'].values);print(r,'rho=%.2f p=%.3g n=%d'%(rho,p,n))
(rho,p),n=sp(t.bv.values,t.La.values);print('adaptive','rho=%.2f p=%.3g n=%d'%(rho,p,n))
print(t[['model','bv','dim','L0.5','L0.33','L0.25','L0.1','La','f16']].round(3).sort_values('L0.1').to_string())
