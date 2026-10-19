import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
from load import *
S=s17()
L=['float32','int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)','binary','hierarchical_merge(0.33) > int4','hierarchical_merge(0.33) > int4norm','hierarchical_merge(0.33) > int4norm(mean)']
for m in ['colbert-ir--colbertv2.0','answerdotai--answerai-colbert-small-v1']:
    a=json.load(open(S[('f9b_int4norm_text',m,'scifact')]));b=json.load(open(S[('f13b_text_scale_same_domain',m,'scifact_plus10000')]))
    ra,rb=rows(a),rows(b)
    print('\n',m,'scifact n_docs',a['n_docs'],'->',b['n_docs'],'nq',a['n_queries'],b['n_queries'],'base ndcg@5 %.4f -> %.4f'%(a['baseline']['ndcg@5'],b['baseline']['ndcg@5']))
    print(f"{'label':44}{'ret f9b':>9}{'ret f13b':>9}  CI f9b / CI f13b  overlap")
    for l in L:
        x,y=ra[l],rb[l]; ca,cb=x['retention_ci:ndcg@5'],y['retention_ci:ndcg@5']
        ov=max(ca[0],cb[0])<=min(ca[1],cb[1])
        print(f"{l:44}{x['retention:ndcg@5']:9.4f}{y['retention:ndcg@5']:9.4f}  [{ca[0]:.3f},{ca[1]:.3f}] [{cb[0]:.3f},{cb[1]:.3f}] {ov}")
    for c in ['int4','int4norm','int4(mean)','int4norm(mean)']:
        print('norm ratio_sd',c,round(a['norms'][c]['ratio_sd'],5),round(b['norms'][c]['ratio_sd'],5))
    # paired per-query diff int4norm - float32 per dataset (query-level)
    import numpy as np
    for c in ['int4','int4norm','int4norm(mean)']:
        pa=np.array(a['per_query_ndcg@10'][c])-np.array(a['per_query_ndcg@10']['float32']);pb=np.array(b['per_query_ndcg@10'][c])-np.array(b['per_query_ndcg@10']['float32'])
        print('ndcg@10 mean delta vs f32',c,'%.4f (se %.4f) -> %.4f (se %.4f)'%(pa.mean(),pa.std(ddof=1)/len(pa)**.5,pb.mean(),pb.std(ddof=1)/len(pb)**.5))
