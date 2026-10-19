from common import *
import pandas as pd
from scipy import stats
M=measured(); rows=[]
for mid,d in M.items():
    if 'vidore/docvqa_test_subsampled' in d:
        rows.append(dict(model=mid,doc=d['vidore/docvqa_test_subsampled']['baseline'],info=d['vidore/infovqa_test_subsampled']['baseline'],
           note=d['vidore/docvqa_test_subsampled'].get('note','')))
x=pd.DataFrame(rows); x['fp16']=x.note.str.contains('float16')
# robust: regress doc on info with logit(info) (ceiling), leave-one-out residual
lg=lambda p:np.log(p/(1-p))
X=lg(x['info'].values); Y=x['doc'].values
res=[]
for i in range(len(x)):
    mk=np.arange(len(x))!=i
    s,b,_,_,_=stats.linregress(X[mk],Y[mk]); pred=s*X[i]+b
    sd=np.std(Y[mk]-(s*X[mk]+b),ddof=2)
    res.append((Y[i]-pred)/sd)
x['z_loo']=res
r=stats.spearmanr(x['doc'],x['info']); print('n',len(x),'spearman doc~info',r)
print(x.sort_values('z_loo')[['model','doc','info','fp16','z_loo']].round(3).head(6).to_string()); print(x.sort_values('z_loo')[['model','doc','info','fp16','z_loo']].round(3).tail(6).to_string())
print('|z|>2:',(abs(x.z_loo)>2).sum(), 'fp16 mean z',x[x.fp16].z_loo.mean().round(2),'fp32 mean z',x[~x.fp16].z_loo.mean().round(2),
   'MW p',stats.mannwhitneyu(x[x.fp16].z_loo,x[~x.fp16].z_loo).pvalue.round(3), 'n fp16',x.fp16.sum())
# text: baseline vs published? registry cards: answerai 74.77 card
