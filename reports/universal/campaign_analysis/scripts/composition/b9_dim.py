import sys,os;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
from scipy.stats import spearmanr
from optivision.adapters import MODELS
info={m.model_id:m for m in MODELS.values()} if isinstance(MODELS,dict) else {m.model_id:m for m in MODELS}
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
df=load(); vpd=df[df.label=='float32'].set_index(['model','ds']).vpd
A=P[(P.pred_type=='all_singles')&(P.kinds=='merge>codec')].copy(); A['err']=A.act-A.pred
A['codec']=A.stages.map(lambda s:s[1]); A=A[A.codec.isin(['binary','int4','lloyd2(7)'])]
A['dim']=[info[m].dimension if m in info else np.nan for m in A.model]
A['vpd']=[vpd[(m,d)] for m,d in zip(A.model,A.ds)]
A['family']=[info[m].family if m in info else '?' for m in A.model]
print(A.dim.isna().sum())
# per model-ds mean err over 3 codecs x 2 merges
u=A.groupby(['model','ds']).agg(err=('err','mean'),dim=('dim','first'),vpd=('vpd','first'),family=('family','first')).reset_index()
u['grp']=np.where(u.ds=='scifact','text','image')
for gr,g in u.groupby('grp'):
    print(gr,len(g),'spearman(err,dim)=%.2f (p=%.2g) spearman(err,vpd)=%.2f (p=%.2g)'%(*spearmanr(g.err,g.dim),*spearmanr(g.err,g.vpd)), 'dim range',g.dim.min(),g.dim.max(),'vpd range',g.vpd.min(),g.vpd.max())
print(u.groupby('grp').err.describe().round(4))
print(u[u.grp=='text'].sort_values('err')[['model','err','dim','vpd']].to_string())
print(u[u.grp=='image'].sort_values('err').head(8)[['model','ds','err','dim','vpd']].to_string())
