import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
P['err']=P.act-P.pred
P['codec']=P.stages.map(lambda s: s[-1] if kind(s[-1])=='codec' else '-')
P['merge']=P.stages.map(lambda s: s[0] if kind(s[0])=='merge' else '-')
P['proj']=P.stages.map(lambda s: [x for x in s if kind(x)=='project'][0] if any(kind(x)=='project' for x in s) else '-')
# baselines: alternative predictors for all_singles
A=P[P.pred_type=='all_singles'].copy()
A['min_pred']=A.parts.map(min)
A['add_pred']=A.parts.map(lambda p:1-sum(1-x for x in p))
A['first_only']=A.parts.map(lambda p:p[0])
for nm in ['pred','min_pred','add_pred','first_only']:
    e=(A.act-A[nm]);print(nm,'MAE %.4f bias %.4f RMSE %.4f'%(e.abs().mean(),e.mean(),np.sqrt((e**2).mean())))
print('--- by kinds x codec, all_singles')
def s(g):
    e=g.act-g.pred; return pd.Series(dict(n=len(g),MAE=e.abs().mean(),bias=e.mean(),worst=e.loc[e.abs().idxmax()],pct_sub=(e<-0.005).mean(),pct_super=(e>0.005).mean(),mean_loss_pred=(1-g.pred).mean()))
print(A.groupby(['kinds','codec']).apply(s).round(4))
print('--- merge>codec by merge ratio')
print(A[A.kinds=='merge>codec'].groupby(['merge','codec']).apply(s).round(4))
