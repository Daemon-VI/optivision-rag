import json,collections
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
c=collections.Counter()
for m,ds in M.items():
    for d,v in ds.items():
        for r in v['rows']:
            if len(r['pipeline']['stages'])==1: c[r['label']]+=1
print(len(M));print(c.most_common(60))
m=next(iter(M));d=next(iter(M[m]));print(m,d,M[m][d]['rows'][0])
