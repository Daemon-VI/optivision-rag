exec(open(r'C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/verify/families-2/an.py').read().split("common=None")[0])
import collections
c=collections.Counter()
for k in img:
    if fams[k] is None: continue
    for d in DS:
        for l in rows(k,d): c[l]+=1
print(len([k for k in img if fams[k]]))
for l,n in c.items():
    if n<68: print(n,l)
for k in img:
  if fams[k]: 
    print(k, len(rows(k,DS[0])), len(rows(k,DS[1])), img[k][DS[0]]['note'][:40])
