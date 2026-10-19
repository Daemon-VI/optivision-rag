import numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
rng=np.random.default_rng(1)
w=pd.read_csv('wide_ret.csv')
w['loss']=1-w.binary
w['ldim']=np.log2(w.dim); w['lvpd']=np.log10(w.vpd)
print(w.groupby('dim').agg(n=('loss','size'),models=('model','nunique'),mean=('loss','mean'),med=('loss','median'),mx=('loss','max')).round(4))
print(w.groupby('ds').loss.describe().round(4))
print(w.groupby('family').agg(n=('loss','size'),models=('model','nunique'),mean=('loss','mean'),med=('loss','median'),mn=('loss','min'),mx=('loss','max'),dims=('dim',lambda s:sorted(set(s)))).round(4).sort_values('mean'))
def boot_spear(x,y,groups,B=2000):
    ug=np.unique(groups); idxs={g:np.where(groups==g)[0] for g in ug}
    out=[]
    for _ in range(B):
        p=rng.choice(ug,len(ug)); i=np.concatenate([idxs[g] for g in p])
        out.append(stats.spearmanr(x[i],y[i])[0])
    return np.nanpercentile(out,[2.5,97.5])
tests=0
print("== Spearman of binary loss vs predictors (model-cluster bootstrap CI)")
def rep(name,sub):
    global tests
    for v in ['ldim','lvpd','base']:
        r,p=stats.spearmanr(sub[v],sub.loss); tests+=1
        ci=boot_spear(sub[v].values,sub.loss.values,sub.model.values)
        print(f"{name:28s} n={len(sub):3d} {v:5s} rho={r:+.3f} p={p:.3g} boot95=[{ci[0]:+.2f},{ci[1]:+.2f}]")
rep('all',w)
rep('image only',w[w.ds!='scifact']); rep('text only',w[w.ds=='scifact'])
rep('docvqa',w[w.ds=='docvqa']); rep('infovqa',w[w.ds=='infovqa'])
rep('image, dim==128 only',w[(w.ds!='scifact')&(w.dim==128)])
rep('image, dim>128 only',w[(w.ds!='scifact')&(w.dim>128)])
# dataset effect within image models both datasets: paired
im=w[w.ds!='scifact'].pivot_table(index='model',columns='ds',values='loss').dropna()
d=im.docvqa-im.infovqa
print("paired docvqa-infovqa binary loss: n",len(d),"mean",d.mean().round(4),"median",d.median().round(4),"wilcoxon p",stats.wilcoxon(d).pvalue,"frac docvqa worse",(d>0).mean().round(3))
bs=[rng.choice(d.values,len(d)).mean() for _ in range(5000)]; print("boot CI",np.percentile(bs,[2.5,97.5]).round(4)); tests+=1
# OLS with cluster bootstrap
def ols(X,y): 
    b,*_=np.linalg.lstsq(X,y,rcond=None); return b
def design(sub,vars_):
    X=[np.ones(len(sub))]+[ (sub[v]-sub[v].mean()).values/sub[v].std() for v in vars_]; return np.column_stack(X)
for name,sub in [('image',w[w.ds!='scifact']),('text',w[w.ds=='scifact']),('all+textflag',w.assign(text=(w.ds=='scifact').astype(float)))]:
    vars_=['ldim','lvpd','base']+(['text'] if name=='all+textflag' else [])
    if name=='text': vars_=['ldim','lvpd','base']
    y=np.log(sub.loss.clip(lower=1e-3)).values  # log loss
    X=design(sub,vars_); b=ols(X,y)
    models=sub.model.values; ug=np.unique(models); idx={g:np.where(models==g)[0] for g in ug}
    bb=[]
    for _ in range(2000):
        p=rng.choice(ug,len(ug)); i=np.concatenate([idx[g] for g in p]); 
        bb.append(ols(X[i],y[i]))
    bb=np.array(bb); res=y-X@b; r2=1-res.var()/y.var()
    print(f"OLS log(loss) {name} n={len(sub)} R2={r2:.2f}")
    for k,nm in enumerate(['const']+vars_): print(f"   {nm:6s} {b[k]:+.3f} [{np.percentile(bb[:,k],2.5):+.2f},{np.percentile(bb[:,k],97.5):+.2f}]")
    tests+=len(vars_)
print("tests so far",tests)
