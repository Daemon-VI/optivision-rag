import json,glob,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup'
def ld(b,m,n): return json.load(open(f'{R}/{b}/{m}/int4norm/{n}.json'))
for m,n in [('colbert-ir--colbertv2.0','colbertv2.0'),('answerdotai--answerai-colbert-small-v1','answerai-colbert-small-v1')]:
    a=ld('f9b_int4norm_text',m,n+'_scifact'); b=ld('f13b_text_scale_same_domain',m,n+'_scifact_plus10000')
    print(m,a['n_docs'],b['n_docs'],a['n_queries'],b['n_queries'],a['baseline']['ndcg@5'],b['baseline']['ndcg@5'])
    ra={r['label']:r for r in a['rows']};rb={r['label']:r for r in b['rows']}
    for l in ['int4','int4norm','int4norm(mean)','binary']:
        x,y=ra[l],rb[l];ca,cb=x['retention_ci:ndcg@5'],y['retention_ci:ndcg@5']
        print(' ',l,'%.3f %.3f'%(x['retention:ndcg@5'],y['retention:ndcg@5']),'overlap',max(ca[0],cb[0])<=min(ca[1],cb[1]))
    for l in ['int4','int4norm']:
        print('  norms',l,a['norms'][l].get('ratio_sd'),b['norms'][l].get('ratio_sd'))
    pa,pb=a['per_query_ndcg@10'],b['per_query_ndcg@10']
    for l in ['int4','int4norm','int4norm(mean)']:
        da=np.array(pa[l])-np.array(pa['float32']);db=np.array(pb[l])-np.array(pb['float32'])
        print('  d10',l,'%.3f %.3f'%(da.mean(),db.mean()), 'se %.3f %.3f'%(da.std(ddof=1)/len(da)**.5,db.std(ddof=1)/len(db)**.5))
# f3 distractors
import os
for p in glob.glob(f'{R}/f3_text_scale/**/run.json',recursive=True)[:1]+glob.glob(f'{R}/f13_text_scale_same_domain/**/run.json',recursive=True)[:1]:
    s=open(p).read(); import re
    print(p[-60:],re.findall(r'distractors[^,\]]*\]?',s)[:3])
