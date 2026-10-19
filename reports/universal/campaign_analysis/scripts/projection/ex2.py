import json,glob
f=glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json')
print(len(f))
d=json.load(open(f[0]))
print(d.keys()); print(d['baseline']); 
for r in d['rows']:
    if r['group'] in('dim','dim+codec'): print(r['group'],r['label'],r['dim'],r['pipeline'],r.get('ndcg@5'),r['retention:ndcg@5'],r['retention_ci:ndcg@5'])
