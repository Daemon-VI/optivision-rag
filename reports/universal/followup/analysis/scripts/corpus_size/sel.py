import json,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
P=[('colbertv2','campaign/batches/11_text/orphans/colbertv2.0/selection_rules/colbertv2.0_scifact.json','followup/f3_text_scale/colbert-ir--colbertv2.0/selection_rules/colbertv2.0_scifact_plus10000.json'),
('answerai','followup/f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/selection_rules/answerai-colbert-small-v1_scifact.json','followup/f3_text_scale/answerdotai--answerai-colbert-small-v1/selection_rules/answerai-colbert-small-v1_scifact_plus10000.json'),
('colSmol','followup/f4_pages_128d/vidore--colSmol-500M/selection_rules/colSmol-500M_tabfquad_test_subsampled.json','followup/f6_scale_colsmol/vidore--colSmol-500M/selection_rules/colSmol-500M_docvqa_test_subsampled_plus2000.json'),
('colpali','campaign/vidore--colpali-v1.3/selection_rules/colpali-v1.3_docvqa_test_subsampled.json','followup/f7_scale_colpali/vidore--colpali-v1.3/selection_rules/colpali-v1.3_docvqa_test_subsampled_plus2000.json')]
for n,a,b in P:
    print('\n==',n)
    for tag,f in (('small',a),('large',b)):
        S=json.load(open(R+f)); print(tag,S['n_docs'],S['n_queries'],'cands',S['n_candidates'])
        for r in S['results']:
            if r['reference']!='labels': continue
            c=collections.Counter(o['selected'] for o in r['outcomes'])
            print('  %-26s t=%.2f met=%.2f hmean=%.3f hmin=%.3f comp=%.1f  %s'%(r['rule'],r['target'],r['met'],r['holdout_mean'],r['holdout_min'],r['compression_median'],dict(c.most_common(2))))
print('\n#### COMPACT')
import numpy as np
for n,a,b in P:
    if n=='colSmol': continue
    row={}
    for tag,f in (('s',a),('L',b)):
        S=json.load(open(R+f))
        for r in S['results']:
            if r['reference']!='labels': continue
            row.setdefault((r['rule'],r['target']),{})[tag]=(r['met'],r['holdout_min'],r['compression_median'])
    print(n)
    for k,v in row.items():
        if 's' in v and 'L' in v: print('  %-26s %.2f  met %.2f->%.2f  hmin %.3f->%.3f comp %.1f->%.1f'%(k[0],k[1],v['s'][0],v['L'][0],v['s'][1],v['L'][1],v['s'][2],v['L'][2]))
    for g in ['calibrate() default','point']:
        v=[row[k] for k in row if k[0]==g and 's' in row[k] and 'L' in row[k]]
        print(' MEAN',g,'met',np.mean([x['s'][0] for x in v]),np.mean([x['L'][0] for x in v]))
