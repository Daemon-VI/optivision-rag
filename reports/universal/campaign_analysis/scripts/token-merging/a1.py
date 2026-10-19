import warnings;warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from scipy import stats
rng=np.random.default_rng(0)
W=pd.read_pickle('W.pkl')
W['kf']={}.get(0,0)
R=[0.5,0.33,0.25,0.1]
for r in R: W[f'k{r}']=W[f'v{r}']/W.bv   # achieved keep fraction
pd.set_option('display.width',200)
def boot_ci(x,f=np.mean,B=5000):
    x=np.asarray(x);x=x[~np.isnan(x)]
    b=[f(rng.choice(x,len(x))) for _ in range(B)];return np.percentile(b,[2.5,97.5])
print('achieved keep fraction')
print(W.groupby('ds')[[f'k{r}' for r in R]].median().round(3))
print('\nRetention by ds and ratio: n, mean, median, 95%boot CI of mean, min, share>=.99,>=.98,>=.95')
for ds,g in W.groupby('ds'):
    for r in R:
        x=g[f'r{r}'].dropna()
        lo,hi=boot_ci(x)
        print(ds,r,len(x),round(x.mean(),4),round(x.median(),4),(round(lo,4),round(hi,4)),round(x.min(),3),round((x>=.99).mean(),2),round((x>=.98).mean(),2),round((x>=.95).mean(),2))
    x=g.a_ret.dropna();print(ds,'adaptive0.6',len(x),round(x.mean(),4),round(x.median(),4),round(x.min(),3),'keepfrac',round((g.a_v/g.bv).median(),3))
# knee: step differences, paired over models
print('\nStep drops (paired, mean diff, bootstrap CI, Wilcoxon p)')
pairs=[(0.5,0.33),(0.33,0.25),(0.25,0.1),(0.5,0.1),(0.5,0.25)]
ntests=0
for ds,g in W.groupby('ds'):
    for a,b in pairs:
        h=g[[f'r{a}',f'r{b}']].dropna();dd=(h[f'r{a}']-h[f'r{b}']).values
        lo,hi=boot_ci(dd); p=stats.wilcoxon(dd).pvalue;ntests+=1
        print(ds,a,'->',b,len(dd),round(dd.mean(),4),(round(lo,4),round(hi,4)),'p=%.2g'%p)
# per-keep-fraction slope: retention loss per unit log2(keep) between segments
print('\nloss per halving of keep (percentage points), by segment, mean over model-datasets')
for ds,g in W.groupby('ds'):
    for a,b in pairs[:3]:
        h=g[[f'r{a}',f'r{b}',f'k{a}',f'k{b}']].dropna()
        halv=np.log2(h[f'k{a}']/h[f'k{b}'])
        print(ds,a,b,round(((h[f'r{a}']-h[f'r{b}'])/halv).mean()*100,3),'pp per halving')
# broken-stick fit on pooled model-ds: ret ~ a + b*max(0,log2k0 - log2 keep)... fit hinge at x=log2(keep) knot c
print('\nPooled hinge fit retention ~ 1 - s*max(0,c - log2(keep)), grid c')
def hinge(g,label):
    xs=[];ys=[]
    for r in R:
        h=g[[f'k{r}',f'r{r}']].dropna();xs+=list(np.log2(h[f'k{r}']));ys+=list(h[f'r{r}'])
    xs=np.array(xs);ys=np.array(ys);best=None
    for c in np.arange(-5,-0.9,0.05):
        z=np.maximum(0,c-xs);b=(z@(1-ys))/(z@z)  # 1-ret = b*z
        sse=((1-ys-b*z)**2).sum()
        if np.isfinite(sse) and (best is None or sse<best[0]): best=(sse,c,b)
    lin_b=np.polyfit(xs,ys,1);sse_lin=((ys-np.polyval(lin_b,xs))**2).sum()
    # bootstrap by model for knot
    ms=g.model.unique();cs=[]
    for _ in range(300):
        s=rng.choice(ms,len(ms));gg=pd.concat([g[g.model==m] for m in s])
        xs2=[];ys2=[]
        for r in R:
            h=gg[[f'k{r}',f'r{r}']].dropna();xs2+=list(np.log2(h[f'k{r}']));ys2+=list(h[f'r{r}'])
        xs2=np.array(xs2);ys2=np.array(ys2);bb=None
        for c in np.arange(-5,-0.9,0.1):
            z=np.maximum(0,c-xs2);b=(z@(1-ys2))/(z@z);sse=((1-ys2-b*z)**2).sum()
            if np.isfinite(sse) and (bb is None or sse<bb[0]): bb=(sse,c)
        cs.append(bb[1])
    print(label,'n',len(xs),'knot log2keep=%.2f (keep=%.3f) slope=%.4f/halving SSE hinge=%.4f vs linear=%.4f; knot boot 95%%: keep %s'%(best[1],2**best[1],best[2],best[0],sse_lin,np.round(2**np.percentile(cs,[2.5,97.5]),3)))
for ds,g in W.groupby('ds'): hinge(g,ds)
