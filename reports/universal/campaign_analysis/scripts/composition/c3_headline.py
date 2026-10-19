import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
P['err']=P.act-P.pred
P['f16']=P.stages.map(lambda s:'float16' in s)
rng=np.random.default_rng(5)
def cb(g,B=2000):
    m=g.groupby('model').err.agg(['sum','count']); s=m['sum'].values;c=m['count'].values;idx=rng.integers(0,len(m),(B,len(m)))
    return s.sum()/c.sum(),*np.percentile(s[idx].sum(1)/c[idx].sum(1),[2.5,97.5])
for pt in ['all_singles','(m>p)x codec']:
    g=P[(P.pred_type==pt)&~P.f16]
    e=g.err
    print(pt,'n',len(g),'models',g.model.nunique(),'MAE %.4f'%e.abs().mean(),'median|e| %.4f'%e.abs().median(),'bias %.4f CI [%.4f,%.4f]'%cb(g),'|e|<.01: %.3f <.02: %.3f <.05: %.3f'%((e.abs()<.01).mean(),(e.abs()<.02).mean(),(e.abs()<.05).mean()),'worst %.4f'%e.loc[e.abs().idxmax()],'RMSE %.4f'%np.sqrt((e**2).mean()),'share err<0 %.3f'%(e<0).mean())
    img=g[g.ds!='scifact'];txt=g[g.ds=='scifact']
    print('  image MAE %.4f bias %.4f n %d | text MAE %.4f bias %.4f n %d'%(img.err.abs().mean(),img.err.mean(),len(img),txt.err.abs().mean(),txt.err.mean(),len(txt)))
# controls: float16/int8 combined
g=P[(P.pred_type=='all_singles')&P.stages.map(lambda s:s[-1] in('float16','int8(per_vector)'))]
print('near-no-op codec controls n',len(g),'MAE %.5f'%g.err.abs().mean(),'p95 %.4f'%g.err.abs().quantile(.95),'in-CI %.3f'%((g.pred>=g.alo)&(g.pred<=g.ahi)).mean())
# float16-baseline subset
g=P[(P.pred_type=='all_singles')&~P.f16]; f=g.note.str.contains('float16 inference')
print('float16-inference baseline rows',f.sum(),'MAE %.4f vs others %.4f'%(g[f].err.abs().mean(),g[~f].err.abs().mean()),'bias %.4f vs %.4f'%(g[f].err.mean(),g[~f].err.mean()))
# retention>1 rows
print('rows with act>1:',(P.act>1).mean())
