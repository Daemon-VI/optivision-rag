from common import *
import pandas as pd, re
from scipy import stats
df=pd.read_pickle(OUT+'/wide_long.pkl'); d=df[df.m=='ndcg@5'].set_index(['model','ds','label'])
rec=[]
for (m,ds,l),r in d.iterrows():
    st=l.split(' > ')
    if len(st)<2 or st[-1]!='binary': continue
    p=' > '.join(st[:-1]); 
    if (m,ds,p) not in d.index: continue
    q=d.loc[(m,ds,p)]
    k=re.search(r'project\((\d+)',p); 
    rec.append(dict(model=m,ds=ds,k=int(k.group(1)) if k else 0,dim=None,r_ext=r.ret,r_pre=q.ret,diff=r.ret-q.ret,lo_ext=r.lo,hi_pre=q.hi,merge=('merge' in p)))
x=pd.DataFrame(rec)
dims=pd.read_pickle(OUT+'/comp.pkl').groupby('model').b_dim.first(); x['dim']=x.model.map(dims); x['frac']=x.k/x.dim
x['kind']=np.where(x.k==0,'no project',np.where(x.frac<=0.25,'k/d<=1/4','k/d>1/4'))
g=x.groupby('kind').agg(n=('diff','size'),mean_diff=('diff','mean'),med=('diff','median'),frac_up=('diff',lambda s:(s>0.01).mean()),sep=('lo_ext',lambda s:0))
x['sep']=x.lo_ext>x.hi_pre
print(x.groupby('kind').agg(n=('diff','size'),mean_diff=('diff','mean'),med=('diff','median'),frac_up_001=('diff',lambda s:(s>0.01).mean()),frac_CI_separated=('sep','mean')).round(3).to_string())
print(x[x.kind=='k/d<=1/4'].groupby('k').agg(n=('diff','size'),mean_diff=('diff','mean'),pre=('r_pre','mean'),ext=('r_ext','mean')).round(3).to_string())
# bootstrap over models for the low-k/d group
lo=x[x.kind=='k/d<=1/4']; mm=lo.groupby('model')['diff'].mean(); rng=np.random.default_rng(2)
bs=[rng.choice(mm.values,len(mm)).mean() for _ in range(5000)]
print('low k/d: models',len(mm),'mean per-model diff',mm.mean().round(3),'CI',np.percentile(bs,[2.5,97.5]).round(3),'models with positive mean',(mm>0).sum())
print('wilcoxon over models p=',stats.wilcoxon(mm.values).pvalue)
# per-model where pre<0.5: how many binary rows beat
print('rows with float project prefix <0.5 retention:',(lo.r_pre<0.5).sum(),'of which binary better by >0.05:',((lo.r_pre<0.5)&(lo['diff']>0.05)).sum())
print(lo.groupby(lo.model.map(lambda s:'text' if s in set(df[df.ds=="scifact"].model) else 'image')).agg(n=('diff','size'),mean_diff=('diff','mean')).round(3))
x.to_pickle(OUT+'/binary_gain.pkl')
