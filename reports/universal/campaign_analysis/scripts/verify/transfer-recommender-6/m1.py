import json,collections
P="C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"
d=json.load(open(P))["models"]
c=collections.Counter()
for m,v in d.items():
    for ds,x in v.items():
        for r in x['rows']: c[r['label']]+=1
print(len(c)); 
for k,n in c.most_common(70): print(n,k)
x=d['vidore/colpali']['vidore/docvqa_test_subsampled']; print(x['baseline'],x['rows'][0])
