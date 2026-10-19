import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300); pd.set_option('display.max_colwidth',60)
A=P[(P.pred_type=='all_singles')].copy()
A['err']=A.act-A.pred
print(A.reindex(A.err.abs().sort_values(ascending=False).index).head(25)[['model','ds','label','act','pred','err','parts']].to_string())
# share of absolute error from top models
A['ae']=A.err.abs()
g=A.groupby('model').ae.sum().sort_values(ascending=False)
print((g/g.sum()).head(10).round(3))
print('top10 errors share of total abs err', A.ae.sort_values(ascending=False).head(50).sum()/A.ae.sum())
