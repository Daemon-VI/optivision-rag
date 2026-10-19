"""Build the master table: one row per (model, dataset). Sources: reports/universal/campaign/*/wide + wide/ (full 4-metric data),
measured.json (fills the 7 older datasets, labels 'none > int4' etc), registry (family, dim, evidence), run.json (dtype), hubrank."""
import json,glob,re,pandas as pd,numpy as np
from optivision.adapters import MODELS
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
HUB={x['id']:x for x in json.load(open('C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json'))}
fix={'colqwen3.5-4.5B-v3':'athrael-soju/colqwen3.5-4.5B-v3','nemotron-colembed-vl-4b-v2':'nvidia/nemotron-colembed-vl-4b-v2','nemotron-colembed-vl-8b-v2':'nvidia/nemotron-colembed-vl-8b-v2','tomoro-colqwen3-embed-4b':'TomoroAI/tomoro-colqwen3-embed-4b'}
w=pd.read_pickle('wide.pkl')
for i,r in w.iterrows():
    if pd.isna(r.model):
        w.at[i,'model']=fix[r.short.split('_')[0]]
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
L=['float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary']
rows=[]
seen=set()
for _,r in w.iterrows():
    d=dict(model=r.model,dataset=('docvqa' if 'docvqa' in r.dataset else 'infovqa' if 'infovqa' in r.dataset else 'scifact'),src='wide',n_docs=r.n_docs,n_q=r.n_q,vpd=r.vpd,base5=r.base5,base10=r.base10)
    for l in L:
        for m in ['ndcg@5','ndcg@10','recall@1','recall@5']:
            d[f'{l}|{m}']=r[f'{l}|{m}']
        ci=r[f'{l}|ci']; d[f'{l}|lo']=ci[0] if ci else np.nan; d[f'{l}|hi']=ci[1] if ci else np.nan
    rows.append(d); seen.add((r.model,d['dataset']))
old={'int4':'none > int4','binary':'none > binary','int8(per_vector)':'none > int8/vec','float16':'none > float16'}
for m,ds in M.items():
    for dn,v in ds.items():
        k=(m,dn.split('/')[-1].replace('_test_subsampled',''))
        if k in seen: continue
        d=dict(model=m,dataset=k[1],src='measured.json',n_docs=v['n_docs'],n_q=v['n_queries'],base5=v['baseline'])
        for x in v['rows']:
            for l,ol in old.items():
                if x['label']==ol or x['label']==l: d[f'{l}|ndcg@5']=x['retention']; d[f'{l}|lo']=x['retention_interval'][0]; d[f'{l}|hi']=x['retention_interval'][1]
            if x['label']=='int4(mean)': d['int4(mean)|ndcg@5']=x['retention']
            if x['label']=='float32' or x['label']=='none > float32': d['vpd']=x['vectors_per_doc']
        rows.append(d)
df=pd.DataFrame(rows)
def reginfo(m):
    mi=MODELS[m]; ev=mi.evidence or ''
    dt='float16-inference' if re.search('float16 inference',ev) else ('float32' if re.search('float32',ev) else 'unknown/auto')
    return pd.Series(dict(family=mi.family,modality=mi.modality,dim=mi.dimension,dtype=dt))
df=pd.concat([df,df.model.apply(reginfo)],axis=1)
df['downloads']=df.model.map(lambda m:HUB.get(m,{}).get('downloads',np.nan))
df['int4']=df['int4|ndcg@5']; df['int4m']=df['int4(mean)|ndcg@5']; df['bin']=df['binary|ndcg@5']; df['int8']=df['int8(per_vector)|ndcg@5']; df['l2']=df['lloyd2(7)|ndcg@5']
df.to_pickle('master2.pkl'); df.to_csv('master2.csv',index=False)
print(len(df),df.model.nunique()); print(df.groupby('modality').model.nunique()); print(df.groupby('dtype').model.nunique())
print(df.src.value_counts())
print(df[df.int4m.isna()|df.int4.isna()][['model','dataset','src','int4','int4m']])
