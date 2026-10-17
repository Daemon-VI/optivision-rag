from load import *
from collections import Counter
print(len(files),len(recs))
c=Counter()
for k,v in recs.items():
    for f,d in v:
        for r in d['rows']:
            if re.fullmatch(r'project\(\d+(, random)?\)',r['label']): c[(d['baseline']['dim'],r['label'])]+=1
for k in sorted(c): print(k,c[k])
print([(k,[x[0][-60:] for x in v]) for k,v in recs.items() if len(v)>1][:6])
print(len({k[0] for k in recs}))
