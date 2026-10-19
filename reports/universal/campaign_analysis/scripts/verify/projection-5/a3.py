import os,json,glob,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
c=collections.Counter()
for f in glob.glob(R+'*/wide/*.json'):
    j=json.load(open(f))
    for r in j['rows']:
        if 'project' in r['label'] and ' > ' not in r['label']: c[(r['label'],r['group'])]+=1
print(c)
