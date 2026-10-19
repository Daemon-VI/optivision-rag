import sys;sys.path.insert(0,'.')
from load import load,R
import json,glob,numpy as np
D=load()
for m,(a,b) in list(D.items())[:8]:
    print(m.split('--')[1][:22],'vpd',round(a['baseline']['vectors_per_doc']),round(b['baseline']['vectors_per_doc']),'r@1',round(a['baseline']['recall@1'],2),round(b['baseline']['recall@1'],2))
q=json.load(open(R+'campaign/colbert-ir--colbertv2.0/colbertv2.0_scifact_qrels.json'));n=json.load(open(R+'followup/f1_text_nfcorpus_a/colbert-ir--colbertv2.0/colbertv2.0_nfcorpus_qrels.json'))
for nm,d in(('scifact',q),('nfcorpus',n)):
    c=[len(v) for v in d.values()]; print(nm,len(d),'rel/query mean',np.mean(c),'median',np.median(c),'p90',np.percentile(c,90),'frac1',np.mean(np.array(c)==1))
    print(str(list(d.values())[0])[:80])
# harshness vs baseline across 8 models (merge .25, binary)
import pandas as pd
from scipy.stats import spearmanr
df=pd.read_pickle('df.pkl'); 
base={m.split('--')[1]:(a['baseline']['ndcg@10'],b['baseline']['ndcg@10']) for m,(a,b) in D.items()}
for l in['hierarchical_merge(0.25)','binary','int4(mean)']:
    x=df[df.label==l].copy(); x['bn']=x.m.map(lambda k:base[k][1]); x['bs']=x.m.map(lambda k:base[k][0])
    print(l,'rho(d, nfc baseline)',round(spearmanr(x.d,x.bn)[0],2),'rho(d,sf baseline)',round(spearmanr(x.d,x.bs)[0],2))
