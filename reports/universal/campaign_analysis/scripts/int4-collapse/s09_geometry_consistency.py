import json,glob,pandas as pd,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
df=pd.read_pickle('master2.pkl')
# --- geometry: 7 datasets, mean norm / query cosine to mean vs plain int4 loss
m={'E2-colpali-docvqa':('vidore/colpali-v1.3-merged','docvqa'),'E2-colpali-infovqa':('vidore/colpali-v1.3-merged','infovqa'),
 'colqwen2-v1.0-merged_docvqa_test_subsampled':('vidore/colqwen2-v1.0-merged','docvqa'),'colqwen2-v1.0-merged_infovqa_test_subsampled':('vidore/colqwen2-v1.0-merged','infovqa'),
 'colqwen2.5-v0.2_docvqa_test_subsampled':('vidore/colqwen2.5-v0.2','docvqa'),'colqwen2.5-v0.2_infovqa_test_subsampled':('vidore/colqwen2.5-v0.2','infovqa'),
 'text-answerai-colbert-small-v1-scifact':('answerdotai/answerai-colbert-small-v1','scifact')}
rows=[]
for k,(mod,ds) in m.items():
    g=json.load(open(R+f'geometry/{k}.json')); r=df[(df.model==mod)&(df.dataset==ds)].iloc[0]
    rows.append(dict(set=k[:28],mean_norm=g['corpus_mean_norm'],q_cos_mean=g['query_cosine_to_corpus_mean_median'],nn_cos=g['median_nn_cosine_within_doc'],int4=r.int4,binary=r.bin,bin_centred=g['rows']['binary centred']['retention'],bin_centred_qdir_removed=g['rows']['binary centred, query mean direction removed']['retention']))
G=pd.DataFrame(rows); print(G.round(3).to_string())
for c in ['mean_norm','q_cos_mean','nn_cos']:
    for y in ['int4','binary','bin_centred']:
        r=stats.spearmanr(G[c],G[y]); print(f'Spearman {c} ~ {y}: rho={r[0]:.2f} p={r[1]:.3f} (n=7; one text point drives it)')
# Only image-only
gi=G.iloc[:6]
for c in ['mean_norm','q_cos_mean']:
    r=stats.spearmanr(gi[c],gi.int4); print('image-only',c,'~int4',round(r[0],2),round(r[1],3))
# --- dataset consistency: image models with both datasets
p=df[df.modality=='image'].pivot_table(index='model',columns='dataset',values='int4').dropna()
r=stats.spearmanr(p.docvqa,p.infovqa); print('image models both datasets n=',len(p),'Spearman int4 doc~info',round(r[0],2),round(r[1],4))
pp=p.drop(index=[i for i in p.index if 'nemotron' in i]); r=stats.spearmanr(pp.docvqa,pp.infovqa); print('  without nemotron n=',len(pp),round(r[0],2),round(r[1],3))
print('  max |doc-info| among non-nemotron',(pp.docvqa-pp.infovqa).abs().max().round(3),' nemotron:',p[p.index.str.contains('nemotron')].round(3).to_string())
# --- EVIE vs nemotron: same width, same format; compare descriptors
c=df[df.model.str.contains('EVIE|nemotron')][['model','dataset','dim','dtype','vpd','base5','int4','int4|lo','int4|hi','int4m','bin','int8']]
print(c.round(3).to_string())
