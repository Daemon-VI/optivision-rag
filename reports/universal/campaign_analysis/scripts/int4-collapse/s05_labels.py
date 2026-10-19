import json,glob,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
c=collections.Counter(); ex={}
for f in glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'wide/*_test_subsampled.json'):
    d=json.load(open(f))
    for r in d['rows']:
        if 'int4' in r['label'] or 'binary' in r['label'] or 'lloyd' in r['label']: c[(r['group'],r['label'])]+=1
for k,v in sorted(c.items()): print(v,k)
d=json.load(open(R+'wide/nemotron-colembed-vl-4b-v2_docvqa_test_subsampled.json'))
for r in d['rows']:
    if 'int4' in r['label'] or r['group']=='dim+codec' or 'binary' in r['label']: print(r['group'],r['label'],round(r['retention:ndcg@5'],3),r.get('retention_ci:ndcg@5') and [round(x,3) for x in r['retention_ci:ndcg@5']])
