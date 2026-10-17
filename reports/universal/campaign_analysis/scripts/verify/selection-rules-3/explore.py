import json,glob,collections
c=collections.Counter()
for f in glob.glob("*/selection_rules/*.json"):
    d=json.load(open(f))
    for r in d['results']:
        c[(r['reference'],r['rule'],r['target'])]+=1
for k,v in sorted(c.items()):print(k,v)
print(d['n_docs'],d['n_queries'])
