from common import *
import pandas as pd
df=pd.read_pickle(OUT+'/wide_long.pkl'); d=df[df.m=='ndcg@5'].set_index(['model','ds','label'])
pd.set_option('display.width',250)
rec=[]
for (m,ds,l),r in d.iterrows():
    st=l.split(' > ')
    if len(st)<2: continue
    p=' > '.join(st[:-1])
    if (m,ds,p) in d.index:
        q=d.loc[(m,ds,p)]
        rec.append(dict(model=m,ds=ds,ext=l,prefix=p,r_ext=r.ret,r_pre=q.ret,diff=r.ret-q.ret,sepa=(r.lo>q.hi), last=st[-1].split('(')[0]))
x=pd.DataFrame(rec); print(len(x),'extension/prefix pairs')
print("ext > prefix by >0.01:",(x['diff']>0.01).sum()," >0.02:",(x['diff']>0.02).sum()," >0.05:",(x['diff']>0.05).sum())
print(x.groupby('last')['diff'].agg(['count','mean','min','max']).round(4).to_string())
print(x.sort_values('diff',ascending=False).head(15)[['model','ds','ext','prefix','r_ext','r_pre','diff']].round(3).to_string())
# Which last-stage give large gain: int4 on top of merged? e.g. merge then int4 > merge alone?
big=x[x['diff']>0.02]; print(big.groupby(['last']).size().to_dict()); print(big.groupby('model').size().sort_values(ascending=False).head(8).to_dict())
