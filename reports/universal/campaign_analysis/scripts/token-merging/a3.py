import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(2)
W=pd.read_pickle('W.pkl')
R=[0.5,0.33,0.25,0.1]
for r in R: W[f'L{r}']=1-W[f'r{r}']
W['La']=1-W.a_ret
D=W[W.ds=='docvqa'].set_index('model');I=W[W.ds=='infovqa'].set_index('model')
common=D.index.intersection(I.index);print('models with both',len(common))
def bci(x,B=5000):
    x=np.asarray(x);return np.percentile([rng.choice(x,len(x)).mean() for _ in range(B)],[2.5,97.5])
print('--- DocVQA vs InfoVQA paired (loss docvqa - loss infovqa), models in both')
for c in [f'L{r}' for r in R]+['La']:
    h=pd.concat([D.loc[common,c],I.loc[common,c]],axis=1).dropna();dd=(h.iloc[:,0]-h.iloc[:,1]).values
    npos=(dd>0).sum();sg=stats.binomtest(npos,len(dd)).pvalue
    print(c,'n=%d mean diff=%.4f CI %s wilcoxon p=%.2g, docvqa worse in %d/%d (sign p=%.2g); ratio of mean losses %.2f'%(len(dd),dd.mean(),np.round(bci(dd),4),stats.wilcoxon(dd).pvalue,npos,len(dd),sg,h.iloc[:,0].mean()/h.iloc[:,1].mean()))
print('\n--- Is it just baseline level? corr of loss with baseline ndcg across image cells; docvqa vs infovqa baseline')
print('mean base docvqa %.3f infovqa %.3f'%(D.base.mean(),I.base.mean()))
im=W[W.modality=='image']
for c in ['L0.25','L0.1','La']:
    x=im[['base',c,'ds']].dropna()
    print(c,'spearman(base,loss) pooled image: rho=%.2f p=%.3g'%stats.spearmanr(x.base,x[c])[:2], end=' | ')
    for ds,g in x.groupby('ds'): print(ds,'rho=%.2f p=%.2g'%stats.spearmanr(g.base,g[c])[:2],end=' ')
    print()
print('\n--- model-level consistency: Spearman of loss between DocVQA and InfoVQA across models')
for c in [f'L{r}' for r in R]+['La']:
    h=pd.concat([D.loc[common,c],I.loc[common,c]],axis=1).dropna()
    print(c,'rho=%.2f p=%.2g n=%d'%(*stats.spearmanr(h.iloc[:,0],h.iloc[:,1]),len(h)))
print('\n--- does mild loss predict heavy loss? Spearman across model-ds cells within dataset')
for ds,g in W.groupby('ds'):
    for a,b in [('L0.5','L0.1'),('L0.33','L0.1'),('L0.25','L0.1'),('L0.5','L0.25')]:
        h=g[[a,b]].dropna();print(ds,a,b,'rho=%.2f p=%.2g n=%d'%(*stats.spearmanr(h[a],h[b]),len(h)))
print('\n--- text (SciFact) vs image: DESCRIPTIVE ONLY (modality, dataset, vpd, n_docs, n_q all confounded)')
for c in [f'L{r}' for r in R]+['La']:
    t=W[W.modality=='text'][c].dropna();i=W[W.modality=='image'][c].dropna()
    u=stats.mannwhitneyu(t,i)
    # model-cluster bootstrap difference of medians
    bs=[np.median(rng.choice(t,len(t)))-np.median(rng.choice(i,len(i))) for _ in range(3000)]
    print(c,'text median %.4f (n=%d) image median %.4f (n=%d) diff CI %s MWU p=%.2g'%(t.median(),len(t),i.median(),len(i),np.round(np.percentile(bs,[2.5,97.5]),4),u.pvalue))
print('\nkeep fraction at which median loss reaches 1%/2%/5% (interp on median-loss-vs-keep over models), by ds')
for ds,g in W.groupby('ds'):
    ks=[0.5,0.33,0.25,0.1];med=[g[f'L{r}'].median() for r in ks];mean=[g[f'L{r}'].mean() for r in ks]
    print(ds,'median loss',np.round(med,4),'mean loss',np.round(mean,4))
    for thr in [0.01,0.02,0.05]:
        xs=np.log2([1,0.5,0.33,0.25,0.1]);ys=[0]+med
        kk=None
        for j in range(len(ys)-1):
            if ys[j]<thr<=ys[j+1]: kk=2**(xs[j]+(thr-ys[j])/(ys[j+1]-ys[j])*(xs[j+1]-xs[j]))
        print('   median loss reaches %.0f%% at keep~%s'%(thr*100,None if kk is None else round(kk,3)))
print('\n--- per-model keep fraction needed for loss<=2%/5% (interpolated, log-linear), count of models by bucket')
for ds,g in W.groupby('ds'):
    out={0.02:[],0.05:[]}
    for _,r in g.iterrows():
        ks=[1,0.5,0.33,0.25,0.1];ls=[0]+[r[f'L{x}'] for x in R]
        for thr in out:
            # smallest keep with loss<=thr: scan; use measured points only
            ok=[k for k,l in zip(ks,ls) if (not np.isnan(l)) and l<=thr]
            out[thr].append(min(ok))
    for thr,v in out.items(): print(ds,'loss<=%.0f%%'%(thr*100),'smallest measured keep achieving it:',pd.Series(v).value_counts().sort_index().to_dict())
