import json,numpy as np
from collections import Counter
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
units={(m,ds):{r['label']:r for r in v['rows']} for m,x in d.items() for ds,v in x.items()}
base={(m,ds):v.get('note','') for m,x in d.items() for ds,v in x.items()}
c=Counter(l for u in units.values() for l in u)
cand=[l for l,n in c.items() if n==88]
print(len(cand))
ev=[k for k,u in units.items() if all(l in u for l in cand)]
print(len(ev),Counter(k[1][:8] for k in ev))
def fam(m): return m.split('/')[-1].lower()
def run(t,scope,drop_extreme=None,exclude_sib=False,lower=False,margin=0.0):
    out=[]
    for k in ev:
        m,ds=k
        peers=[p for p in ev if p[0]!=m and (scope=='all' or p[1]==ds)]
        if exclude_sib:
            org=m.split('/')[1].lower()[:6]
            peers=[p for p in peers if p[0].split('/')[1].lower()[:6]!=org]
        if drop_extreme: peers=[p for p in peers if p[0]!=drop_extreme]
        ok=[l for l in cand if all((units[p][l]['retention_interval'][0] if lower else units[p][l]['retention'])>=t+margin for p in peers)]
        best=max(ok,key=lambda l:units[k][l]['compression'])  # choose by held-out? no: compression is peer-independent mostly
        # choose by median peer compression instead
        best=max(ok,key=lambda l:np.median([units[p][l]['compression'] for p in peers]))
        r=units[k][best]
        own=[units[k][l]['compression'] for l in cand if units[k][l]['retention']>=t]
        ob=max(own)
        out.append((k,best,r['retention']>=t,r['compression'],ob,len(peers)))
    return out
for scope in['all','same']:
  for t in[.95,.97,.99]:
    o=run(t,scope)
    hit=np.mean([x[2] for x in o]);comp=np.median([x[3] for x in o]);ob=np.median([x[4] for x in o])
    fr=np.median([x[3]/x[4] for x in o if x[2]])
    print(scope,t,len(o),round(hit,3),round(comp,1),round(ob,1),round(fr,3),Counter(x[1] for x in o).most_common(3),[x[0] for x in o if not x[2]])
