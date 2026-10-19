import json,collections
m=json.load(open('measured.json'))['models']
c=collections.Counter()
for mid,ds in m.items():
    for d,v in ds.items():
        for r in v['rows']:
            if 'project' in r['label'] or 'pca' in r['label']: c[r['label']]+=1
print(len(c)); 
for k,v in sorted(c.items())[:80]: print(k,v)
mid=list(m)[0]; d=list(m[mid])[0]; print(mid,d,m[mid][d]['rows'][0])
