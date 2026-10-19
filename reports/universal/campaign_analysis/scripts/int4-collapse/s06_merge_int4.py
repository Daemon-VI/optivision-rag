"""Plain int4 applied after hierarchical_merge (0.25, 0.33): does token pooling change the collapse? Uses wide files, ndcg@5."""
import json,glob,pandas as pd,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
w=pd.read_pickle('wide.pkl')
fixm={'colqwen3.5-4.5B-v3':'athrael-soju/colqwen3.5-4.5B-v3','nemotron-colembed-vl-4b-v2':'nvidia/nemotron-colembed-vl-4b-v2','nemotron-colembed-vl-8b-v2':'nvidia/nemotron-colembed-vl-8b-v2','tomoro-colqwen3-embed-4b':'TomoroAI/tomoro-colqwen3-embed-4b'}
rows=[]
for _,x in w.iterrows():
    d=json.load(open(R+x.file)); lab={r['label']:r for r in d['rows']}
    m=x.model if isinstance(x.model,str) else fixm[x.short.split('_')[0]]
    for q in ['0.25','0.33']:
        a=lab.get(f'hierarchical_merge({q})'); b=lab.get(f'hierarchical_merge({q}) > int4'); c=lab.get(f'hierarchical_merge({q}) > int4(mean)'); e=lab.get(f'hierarchical_merge({q}) > binary')
        rows.append(dict(model=m,dataset=x.dataset[-8:],q=q,plain=lab['int4']['retention:ndcg@5'],merge=a['retention:ndcg@5'],m_int4=b['retention:ndcg@5'],m_int4c=c['retention:ndcg@5'],m_bin=e['retention:ndcg@5'],bin=lab['binary']['retention:ndcg@5'],vpd=a['vectors_per_doc']/x.vpd))
t=pd.DataFrame(rows); t['rel']=t.m_int4/t['merge']
t.to_pickle('merge_int4.pkl')
pd.set_option('display.width',200)
print('rows',len(t))
r=stats.spearmanr(t.plain,t.rel); print('Spearman plain int4 vs merged-plain-int4/merge rel:',round(r[0],3),r[1])
deg=t[(t.plain<0.95)]
print(deg.sort_values(['model','q']).round(3).to_string())
# paired change among degraded, per (model,dataset,q)
dd=deg.assign(delta=deg.rel-deg.plain)
print('degraded rows',len(dd),'rel>plain in',(dd.delta>0).sum(),'median delta',dd.delta.median().round(3))
print(stats.wilcoxon(dd.rel,dd.plain))
# non-degraded: does merging cause plain int4 to degrade anywhere?
nd=t[t.plain>=0.95]; print('non-degraded rows',len(nd),'min rel',nd.rel.min().round(3)); print(nd.sort_values('rel').head(6).round(3).to_string())
