import json,collections
d=json.load(open('src/optivision/presets/measured.json'))['models']
c=collections.Counter()
for m,v in d.items():
    for ds,x in v.items():
        c[len(x['rows'])]+=1
print(sorted(c.items()))
print(collections.Counter(ds for v in d.values() for ds in v))
# pairs with different notes
for m,v in list(d.items())[:3]:
    for ds,x in v.items(): print(m,ds,x['n_docs'],x['n_queries'],x['note'][:100],len(x['rows']))
labs=collections.Counter(r['label'] for v in d.values() for x in v.values() for r in x['rows'])
print(len(labs)); print(labs.most_common(40))
