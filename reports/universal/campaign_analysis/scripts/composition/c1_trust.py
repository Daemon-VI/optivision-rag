import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
from scipy.stats import spearmanr
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P[P.pred_type=='all_singles'].copy(); A['err']=A.act-A.pred
A['maxloss']=A.parts.map(lambda p:1-min(p))
A['bin']=pd.cut(A.maxloss,[-1,0.01,0.02,0.03,0.05,0.1,1.01])
print('=== all_singles: error vs largest single-stage loss')
print(A.groupby('bin').apply(lambda x:pd.Series(dict(n=len(x),models=x.model.nunique(),MAE=x.err.abs().mean(),bias=x.err.mean(),p95abs=x.err.abs().quantile(.95),worst_under=x.err.min(),worst_over=x.err.max(),frac_sub_gt1pt=(x.err<-0.01).mean()))).round(4))
print('rule: max single loss<=0.03 -> share of rows',(A.maxloss<=0.03).mean(),'; share of rows with err<-0.02 under rule',((A.err<-0.02)&(A.maxloss<=0.03)).sum(),'of',(A.err<-0.02).sum())
# 3-stage predictors on matched rows
T=P[P.nst==3]
key=['model','ds','label']
w=T.pivot_table(index=key,columns='pred_type',values='pred').dropna().join(T[T.pred_type=='all_singles'].set_index(key).act)
print('=== 3-stage rows with all three predictors available: n=%d'%len(w))
for c in ['all_singles','(m>p)x codec','merge x (p>c)']:
    e=w.act-w[c];print(c,'MAE %.4f bias %.4f worst %.4f'%(e.abs().mean(),e.mean(),e.loc[e.abs().idxmax()]))
# by k/dim ratio in 3-stage (m>p)xcodec
from optivision.adapters import MODELS
info={m.model_id:m for m in MODELS.values()} if isinstance(MODELS,dict) else {m.model_id:m for m in MODELS}
B=P[(P.pred_type=='(m>p)x codec')].copy(); B['err']=B.act-B.pred
B['k']=B.stages.map(lambda s:int(re.search(r'project\((\d+)\)',s[1]).group(1)))
B['dim']=[info[m].dimension for m in B.model]
B['frac']=B.k/B.dim
B['fbin']=pd.cut(B.frac,[0,0.12,0.26,0.51,1.01])
B['codec']=B.stages.map(lambda s:s[2])
print('=== (m>p)xcodec by projection fraction k/dim')
print(B.groupby('fbin').apply(lambda x:pd.Series(dict(n=len(x),models=x.model.nunique(),MAE=x.err.abs().mean(),bias=x.err.mean(),worst=x.err.min()))).round(4))
print(B.groupby('codec').apply(lambda x:pd.Series(dict(n=len(x),MAE=x.err.abs().mean(),bias=x.err.mean(),worst=x.err.min()))).round(4))
# text vs image for merge>project>codec: pred type (m>p)xcodec
B['grp']=np.where(B.ds=='scifact','text','image')
print(B.groupby(['grp','codec']).apply(lambda x:pd.Series(dict(n=len(x),MAE=x.err.abs().mean(),bias=x.err.mean()))).round(4))
B.to_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/B3.pkl")
