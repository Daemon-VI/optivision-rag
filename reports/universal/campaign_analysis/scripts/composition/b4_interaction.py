import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P[P.pred_type=='all_singles'].copy()
A['err']=A.act-A.pred
A['codec']=A.stages.map(lambda s: s[-1] if kind(s[-1])=='codec' else '-')
A['grp']=np.where(A.ds=='scifact','text/scifact','image/vidore')
rng=np.random.default_rng(0)
def cboot(g,col='err',B=2000):
    # cluster bootstrap over models
    models=g.model.unique(); m=g.groupby('model')[col].agg(['sum','count'])
    s=m['sum'].values;c=m['count'].values;n=len(models)
    est=s.sum()/c.sum()
    idx=rng.integers(0,n,(B,n))
    bs=s[idx].sum(1)/c[idx].sum(1)
    return est,*np.percentile(bs,[2.5,97.5])
rows=[];ntests=0
for (k,c,gr),g in A.groupby(['kinds','codec','grp']):
    if c in('float16',): pass
    e,lo,hi=cboot(g)
    rows.append(dict(kinds=k,codec=c,grp=gr,n=len(g),nmodels=g.model.nunique(),bias=e,lo=lo,hi=hi,MAE=g.err.abs().mean(),sig=(lo>0)|(hi<0)))
R=pd.DataFrame(rows);print(R.round(4).to_string())
R.to_csv(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/b4_bias.csv",index=False)
print('n cells',len(R))
