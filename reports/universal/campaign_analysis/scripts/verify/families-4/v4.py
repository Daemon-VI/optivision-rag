exec(open(__file__.replace('v4.py','v.py')).read().split("for nm,only in")[0])
import collections
img=[k for k in d if reg[k].modality=='image']
c=collections.Counter(); 
for k in img:
  for ds in d[k]: c.update(prof(k,ds).keys())
n=sum(len(d[k]) for k in img); print(n)
print({l:v for l,v in c.items() if v<n})
for k in img:
  for ds in d[k]:
    p=prof(k,ds)
    if 'float32' not in p or 'binary' not in p: print(k,ds,len(p),[l for l in ['float32','binary','lloyd2(7)','hierarchical_merge(0.25)'] if l not in p])
