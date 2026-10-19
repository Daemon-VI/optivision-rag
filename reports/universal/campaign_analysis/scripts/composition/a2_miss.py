import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
import collections
df=load()
idx={(r.model,r.ds,r.label) for r in df.itertuples()}
c=collections.Counter()
for r in df[df.stages.map(len)>1].itertuples():
    for s in r.stages:
        if (r.model,r.ds,s) not in idx: c[s]+=1
print(c.most_common(30))
# singles available per model: which project k
pk=df[df.label.str.fullmatch(r'project\(\d+\)')].groupby(['model','ds']).label.apply(lambda x: sorted(x)).head(10)
print(pk)
