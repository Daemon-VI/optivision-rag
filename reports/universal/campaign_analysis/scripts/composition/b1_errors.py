import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=build_pairs(); P.to_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
P['err']=P.act-P.pred
P['lr']=np.log(np.clip(P.act,1e-3,None)/np.clip(P.pred,1e-3,None))
P['incl']=(P.pred>=P.alo)&(P.pred<=P.ahi)
def summ(g):
    return pd.Series(dict(n=len(g),MAE=g.err.abs().mean(),bias=g.err.mean(),worst=g.err.loc[g.err.abs().idxmax()],med_abs=g.err.abs().median(),p95=g.err.abs().quantile(.95),in_CI=g.incl.mean(),mean_act=g.act.mean(),mean_pred=g.pred.mean()))
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(P.groupby(['pred_type','kinds']).apply(summ).round(4))
