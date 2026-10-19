import numpy as np,pandas as pd
from scipy import stats
from load import load
R,B=load(); rng=np.random.default_rng(2)
ms=['ndcg@5','ndcg@10','recall@1','recall@5']
# symmetric loss-ratio check for (a)
J=pd.read_pickle('J.pkl');M='ndcg@5'
mean_r=(J['ret_'+M+'_d']+J['ret_'+M+'_i'])/2
for lo,hi in [(0,.9),(.9,.97),(0,.97)]:
    h=J[(mean_r>=lo)&(mean_r<hi)]
    lr=(1-h['ret_'+M+'_i'])/(1-h['ret_'+M+'_d']); pm=h.assign(lr=lr).groupby('model').lr.median()
    print('sym select mean-ret in [%.2f,%.2f) n=%d median ratio %.3f; models<1 %d/%d'%(lo,hi,len(h),lr.median(),(pm<1).sum(),len(pm)))
# Also split-half: select on docvqa retention of a *different* model? skip.
print('--- (b) metrics')
R['bin']=pd.cut(R.comp,[0,4,16,64,256,1e5],labels=['<4x','4-16x','16-64x','64-256x','>256x'])
for ds,g in list(R.groupby('ds'))+[('all',R)]:
    print('==',ds,len(g))
    print(g[['ret_'+m for m in ms]].agg(['mean','median']).round(4).to_string())
# per-row ordering: which is lowest, strict
def order(g):
    X=g[['ret_'+m for m in ms]].values
    low=X.argmin(1); tie=(np.isclose(X,X.min(1,keepdims=True),atol=1e-12).sum(1)>1)
    return low,tie
for ds,g in R.groupby('ds'):
    X=g[['ret_'+m for m in ms]]
    sub=g[(X.min(1)<.97)]  # non-saturated rows
    low,tie=order(sub)
    print(ds,'non-saturated rows',len(sub),'lowest metric share (ties incl)',dict(zip(ms,np.bincount(low,minlength=4)/len(sub))), 'ties',tie.mean())
# paired differences with model-clustered bootstrap, restricted to rows with min ret<0.97 (non-saturated)
def clustered(g,a,b,nb=2000):
    d=(g['ret_'+a]-g['ret_'+b]); per=d.groupby(g.model).mean(); 
    bs=[rng.choice(per.values,len(per)).mean() for _ in range(nb)]
    return d.mean(),per.mean(),np.percentile(bs,[2.5,97.5]),(per>0).sum(),len(per),stats.wilcoxon(per).pvalue
pairs=[('recall@1','ndcg@5'),('ndcg@5','ndcg@10'),('recall@5','ndcg@5'),('recall@1','recall@5'),('recall@5','ndcg@10'),('recall@1','ndcg@10')]
tests=0
for ds,g in list(R.groupby('ds')):
    sub=g[g[['ret_'+m for m in ms]].min(1)<.97]
    for a,b in pairs:
        r=clustered(sub,a,b);tests+=1
        print(ds,'%s - %s: row-mean %.4f model-mean %.4f CI(%.4f,%.4f) models a>b %d/%d wilcoxon p=%.2g'%(a,b,r[0],r[1],r[2][0],r[2][1],r[3],r[4],r[5]),'n_rows',len(sub))
print('tests',tests)
# by compression bin, model-mean diffs recall@1 - ndcg@5 and recall@5 - ndcg@5
for ds,g in R.groupby('ds'):
    print('--',ds)
    t=g.groupby('bin',observed=True).apply(lambda x:pd.Series({'n':len(x),'ndcg5':x['ret_ndcg@5'].mean(),'ndcg10':x['ret_ndcg@10'].mean(),'r1':x['ret_recall@1'].mean(),'r5':x['ret_recall@5'].mean()}))
    t['r1-n5']=t.r1-t.ndcg5;t['r5-n5']=t.r5-t.ndcg5;t['n10-n5']=t.ndcg10-t.ndcg5;print(t.round(4))
# group
for ds,g in R.groupby('ds'):
    print('--',ds)
    t=g.groupby('group').apply(lambda x:pd.Series({'n':len(x),'ndcg5':x['ret_ndcg@5'].mean(),'r1-n5':(x['ret_recall@1']-x['ret_ndcg@5']).mean(),'r5-n5':(x['ret_recall@5']-x['ret_ndcg@5']).mean(),'n10-n5':(x['ret_ndcg@10']-x['ret_ndcg@5']).mean()}))
    print(t.round(4))
# noise: CI width per metric
for ds,g in R.groupby('ds'):
    print(ds,'mean CI width',{m:round((g['hi_'+m]-g['lo_'+m]).mean(),4) for m in ms})
# safe sets by metric (lower>=0.95)
for ds,g in R.groupby('ds'):
    print(ds,'share lower>=.95',{m:round((g['lo_'+m]>=.95).mean(),3) for m in ms},'share point>=.95',{m:round((g['ret_'+m]>=.95).mean(),3) for m in ms})
    s={m:g['lo_'+m]>=.95 for m in ms}
    print('  safe on ndcg@5 but not recall@1:',(s['ndcg@5']&~s['recall@1']).sum(),' reverse:',(~s['ndcg@5']&s['recall@1']).sum(),
      '| ndcg@5 not recall@5:',(s['ndcg@5']&~s['recall@5']).sum(),' reverse:',(~s['ndcg@5']&s['recall@5']).sum(),
      '| ndcg@5 not ndcg@10:',(s['ndcg@5']&~s['ndcg@10']).sum(),' reverse:',(~s['ndcg@5']&s['ndcg@10']).sum())
# does ndcg@5 safe imply all metrics safe? Among ndcg@5-safe rows, share where any other metric has lower <.95
for ds,g in R.groupby('ds'):
    sf=g[g['lo_ndcg@5']>=.95]
    print(ds,'ndcg5-safe rows',len(sf),'min other lower bound <.95:',(sf[['lo_'+m for m in ms[1:]]].min(1)<.95).sum(),' point ret of any other <.95:',(sf[['ret_'+m for m in ms[1:]]].min(1)<.95).sum(), 'min other point ret',sf[['ret_'+m for m in ms[1:]]].min().min().round(3))
R.to_pickle('R.pkl')
