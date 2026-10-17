from common import *
import itertools, pandas as pd
M=measured()
# profile: retention per label per (model,dataset) from measured.json (55 models)
prof={}
for mid,d in M.items():
    for ds,v in d.items():
        prof[(mid,ds.split('/')[1][:5])]={r['label']:r['retention'] for r in v['rows']}
        prof[(mid,ds.split('/')[1][:5])]['__b']=v['baseline']
keys=list(prof)
out=[]
for ds in ('docvq','infov','scifa'):
    ks=[k for k in keys if k[1]==ds]
    for a,b in itertools.combinations(ks,2):
        la=set(prof[a])&set(prof[b])-{'__b'}
        if len(la)<30: continue
        x=np.array([prof[a][l] for l in la]); y=np.array([prof[b][l] for l in la])
        out.append((ds,a[0],b[0],len(la),np.abs(x-y).mean(),np.abs(x-y).max(),abs(prof[a]['__b']-prof[b]['__b'])))
df=pd.DataFrame(out,columns=['ds','a','b','n','mad','maxd','dbase'])
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
for ds in ('docvq','infov','scifa'):
    s=df[df.ds==ds].sort_values('mad')
    print(ds,'pairs',len(s),'median MAD',s.mad.median().round(4)); print(s.head(8).round(4).to_string(index=False))
# exact equal baselines between different models
bb={}
for k in keys: bb.setdefault((k[1],prof[k]['__b']),[]).append(k[0])
print({k:v for k,v in bb.items() if len(v)>1})
df.to_pickle(OUT+'/dups.pkl')
