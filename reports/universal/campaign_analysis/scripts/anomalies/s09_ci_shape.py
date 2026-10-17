from common import *
import pandas as pd
df=pd.read_pickle(OUT+'/wide_long.pkl'); d=df[(df.m=='ndcg@5')&(df.w>1e-9)].copy()
d['asym']=(d.hi-d.ret)/(d.ret-d.lo).replace(0,np.nan)
d['pos']=(d.ret-d.lo)/d.w
print(d.pos.describe())
print("point in outer 10% of CI (pos<0.1 or >0.9):",((d.pos<0.1)|(d.pos>0.9)).sum(),"of",len(d))
print(d[(d.pos<0.1)|(d.pos>0.9)].sort_values('pos').head(10)[['model','ds','label','ret','lo','hi','pos']].to_string())
# merge-only sign test
from scipy import stats
for lab in ['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)']:
    g=df[(df.m=='ndcg@5')&(df.label==lab)]
    up=(g.ret>1+1e-9).sum(); dn=(g.ret<1-1e-9).sum()
    print(lab,len(g),'up',up,'dn',dn,'mean',round(g.ret.mean(),4),'p',round(stats.binomtest(up,up+dn,0.5).pvalue,4), 'median comp',round(g.comp.median(),2))
# width vs nq: ratio of widths for the 4 metrics
w=df.pivot_table(index=['model','ds','label'],columns='m',values='w'); print((w.median()).round(4).to_dict())
# calibration of width: the float16 / int8 rows: width should be small
print(df[(df.m=='ndcg@5')&(df.label.isin(['float16','int8(per_vector)']))].groupby(['label','ds']).w.median().round(4).to_dict())
