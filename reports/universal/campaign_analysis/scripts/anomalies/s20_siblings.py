from common import *
import pandas as pd
d=pd.read_pickle(OUT+'/dups.pkl')
pairs=[('vidore/colpali-v1.2','vidore/colpali-v1.2-merged'),('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.2-merged','vidore/colpali-v1.2-hf'),
 ('vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),('vidore/colpali-v1.2','vidore/colpali-v1.3'),('vidore/colpali-v1.1','vidore/colpali-v1.2'),
 ('tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'),
 ('Metric-AI/ColQwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0'),
 ('vidore/colqwen2-v0.1','vidore/colqwen2-v1.0-hf'),('Hcompany/NeoMME-260M-Retriever-ST-late','Hcompany/NeoMME-800M-Retriever-ST-late'),
 ('tencent/EVIE-4.5B','tencent/EVIE-8B'),('lightonai/colbertv2.0','colbert-ir/colbertv2.0'),('lightonai/LateOn','lightonai/GTE-ModernColBERT-v1'),('LiquidAI/LFM2-ColBERT-350M','LiquidAI/LFM2.5-ColBERT-350M'),
 ('mixedbread-ai/mxbai-edge-colbert-v0-17m','mixedbread-ai/mxbai-edge-colbert-v0-32m')]
out=[]
for a,b in pairs:
    for _,r in d.iterrows():
        if (r.a==a and r.b==b) or (r.a==b and r.b==a): out.append((a.split('/')[1][:30],b.split('/')[1][:30],r.ds,int(r.n),round(r.mad,4),round(r.maxd,3),round(r.dbase,4)))
print(pd.DataFrame(out,columns=['a','b','ds','n','mad','maxd','dbase']).to_string(index=False))
unr=d[d.n>=54]; print('all pairs n>=54 median MAD by ds',unr.groupby('ds').mad.median().round(4).to_dict(),' 5th pct',unr.groupby('ds').mad.quantile(0.05).round(4).to_dict())
