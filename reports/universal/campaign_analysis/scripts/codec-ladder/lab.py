import json,glob,collections
R="C:/Users/Lenovo/projects/optivision-package"
c=collections.Counter()
for f in glob.glob(R+"/reports/universal/campaign/*/wide/*.json"):
    d=json.load(open(f))
    for r in d['rows']: c[(r['group'],r['label'])]+=1

for k,v in sorted(c.items()): print(k,v)
