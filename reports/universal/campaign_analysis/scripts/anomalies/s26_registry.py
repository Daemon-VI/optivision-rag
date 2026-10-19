from common import *
from optivision.adapters import MODELS
M=measured()
reg={m.model_id:m for m in MODELS} if not isinstance(MODELS,dict) else MODELS
print(len(reg),'registry entries; status counts',__import__('collections').Counter(m.status for m in reg.values()))
meas_reg={k for k,m in reg.items() if m.status=='measured'}
print('registry measured not in measured.json:',sorted(meas_reg-set(M)))
print('measured.json not measured in registry:',sorted((set(M)-meas_reg)))
# evidence dtype vs outcomes dtype
O=outcomes()
for k,o in O.items():
    if k in reg and o['outcome']=='measured':
        ev=reg[k].evidence.lower(); dt=o['dtype']
        flag = ('float16' in ev and dt=='float32') or ('float32' in ev and dt=='float16')
        if flag: print('DTYPE MISMATCH',k,dt,ev[:160])
# vectors per page claims vs measured baseline
import re
bad=[]
for k,m in reg.items():
    if k in M:
        for ds,v in M[k].items():
            pass
# n models with 'float16 inference' in note
print('fp16-inference models in measured.json:',sum(1 for k,d in M.items() if any('float16' in v.get('note','') for v in d.values())),'of',len(M))
