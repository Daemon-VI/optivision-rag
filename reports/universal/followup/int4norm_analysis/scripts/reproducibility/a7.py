import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
from load import *
S=s17();T=s13()
m=[k for k in S if (k[1],k[2]) in T and any('f3_text' not in s for s,_ in T[(k[1],k[2])])]
print('s17 units',len(S),'with s13 twin (excl f3 FiQA)',len(m),'without',len(S)-len(m))
print([ (k[0][:4],k[1].split('--')[1],k[2]) for k in S if k not in m])
