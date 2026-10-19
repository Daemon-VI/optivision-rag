from common import *
U=load(); print(len(U))
c=collections.Counter(k for u in U for k in u['rows'])
print(len(c))
for k,n in sorted(c.items(),key=lambda x:-x[1]):
    if n>=40: pass
print(sum(1 for n in c.values() if n>=80), sum(1 for n in c.values() if n>=60), sum(1 for n in c.values() if n>=90))
for k,n in sorted(c.items(),key=lambda x:-x[1])[:100]: print(n,key_str(k))
print([ (u['model'],u['ds'][:12],len(u['rows'])) for u in U if len(u['rows'])<45])
print(collections.Counter(u['family'] for u in U))
