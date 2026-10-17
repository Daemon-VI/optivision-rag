from common import *
import pandas as pd
df=pd.read_pickle(OUT+'/wide_long.pkl'); d=df[df.m=='ndcg@5']
pd.set_option('display.width',250)
low=d[d.ret<0.05]
print('rows ret<0.05:',len(low),'models',low.model.nunique())
print(low.groupby(['model','ds']).size().to_string())
print(low.label.str.replace(r'hierarchical_merge\(0\.\d+\) > ','HM > ',regex=True).value_counts().head(12).to_string())
M=measured()
# the older nemotron models in measured.json only
for mid in ['nvidia/nemotron-colembed-vl-4b-v2','nvidia/nemotron-colembed-vl-8b-v2']:
    for ds,v in M[mid].items():
        rows=sorted(v['rows'],key=lambda r:r['retention'])[:6]
        print(mid,ds[-8:],[(r['label'],round(r['retention'],3)) for r in rows])
        print(v['note'])
