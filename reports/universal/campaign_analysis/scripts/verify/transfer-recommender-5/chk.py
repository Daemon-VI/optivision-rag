import json
m=json.load(open('src/optivision/presets/measured.json'))['models']
for k,ds,l in [('Cognitive-Lab/ColNetraEmbed','vidore/docvqa_test_subsampled','hierarchical_merge(0.25) > lloyd2(7)'),('lightonai/LateOn','BeIR/scifact','int4(mean)'),('mixedbread-ai/mxbai-colbert-large-v1','BeIR/scifact',None)]:
    for r in m[k][ds]['rows']:
        if l is None and ('project' in r['label'] and 'int4(mean)' in r['label'] and r['retention']<0.92 and r['retention']>0.88) or r['label']==l: print(k,r['label'],r['retention'],r['retention_interval'])
