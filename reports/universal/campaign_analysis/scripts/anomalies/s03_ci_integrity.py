from common import *
import pandas as pd
W=wide_files(); rec=[]
for (folder,fn),w in W.items():
    mid=folder.replace('--','/',1)
    for r in w['rows']:
        for m in ('ndcg@5','ndcg@10','recall@1','recall@5'):
            lo,hi=r['retention_ci:'+m]; p=r['retention:'+m]
            rec.append(dict(model=mid,ds=fn.split('_',1)[1][:7],label=r['label'],group=r['group'],m=m,ret=p,lo=lo,hi=hi,nq=w['n_queries'],comp=r['compression_vs_float32']))
df=pd.DataFrame(rec); df['w']=df.hi-df.lo
print(len(df),'row-metric CIs')
print("point outside CI:",((df.ret<df.lo-1e-12)|(df.ret>df.hi+1e-12)).sum())
x=df[(df.ret<df.lo-1e-12)|(df.ret>df.hi+1e-12)]; print(x.head(10).to_string())
print("zero-width CI with ret != 1:",((df.w<1e-9)&(abs(df.ret-1)>1e-9)).sum())
print("zero-width, ret==1, non-baseline label:", df[(df.w<1e-9)&(df.label!='float32')].groupby('m').size().to_dict())
d5=df[df.m=='ndcg@5']
print("ndcg@5 rows ret>1:",(d5.ret>1).sum(),"of",len(d5)," ret>1.0001:",(d5.ret>1.0001).sum())
print("lo>1 (CI excludes 1 from above):",(d5.lo>1).sum()," by ds:",d5[d5.lo>1].groupby('ds').size().to_dict())
print(d5[d5.lo>1].sort_values('lo',ascending=False).head(15)[['model','ds','label','ret','lo','hi']].to_string())
print("hi>1 for ndcg@5 (CI above 1):",(d5.hi>1).sum(),"; for recall@1:",(df[df.m=='recall@1'].hi>1).sum())
# max ret by metric
print(df.groupby('m').ret.agg(['max','median']).to_string())
# CI width vs 1-ret (should scale with degradation)
d=d5[d5.group!='codec'].copy()
print(d.groupby('ds').w.describe().to_string())
# intervals with width exactly 0 at ret ==1 on non-baseline: which labels
z=df[(df.w<1e-9)&(df.label!='float32')&(df.m=='ndcg@5')]
print(z.groupby('label').size().sort_values(ascending=False).head(15).to_string())
df.to_pickle(OUT+'/wide_long.pkl')
