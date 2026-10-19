import json,numpy as np,sys
from scipy.stats import spearmanr
sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/src')
from optivision.adapters import MODELS
U=json.load(open('C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/verify/projection-1/units.json'))
# per-model mean loss at k=128 for d>128, and k=64 d=128
mm={}
for u in U:
    c=u['curve']
    if '128' in c and u['d']>128: mm.setdefault(u['model'],[]).append((u['d'],1-c['128'][1]))
rows=sorted((v[0][0],np.mean([x[1] for x in v]),k) for k,v in mm.items())
for r in rows: print(r)
d=[r[0] for r in rows];l=[r[1] for r in rows]
print(spearmanr(d,l))
rng=np.random.default_rng(1)
b=[]
for _ in range(5000):
    i=rng.integers(0,len(d),len(d))
    if len(set(np.array(d)[i]))<2: continue
    b.append(spearmanr(np.array(d)[i],np.array(l)[i])[0])
print(np.nanpercentile(b,[2.5,97.5]))
# 128d models loss at k=64 vs wide models at k=64? wide have k=80 only
