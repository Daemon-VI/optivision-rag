import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
D=load()
models=['vidore--colSmol-500M','vidore--colpali-v1.3','vultr--VultronRetrieverCore-Qwen3.5-4.5B','tencent--EVIE-4.5B']
M='retention:ndcg@5';C='retention_ci:ndcg@5'
for k,v in sorted(D.items()):
    if k[0] in models or 'qwen2.5-v0.2' in k[0]: print(k,v['nd'],v['nq'],round(v['base'],3),len(v['rows']))
