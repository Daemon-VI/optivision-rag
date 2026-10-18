import sys;sys.path.insert(0,'.')
from load import load
import numpy as np
from scipy.stats import spearmanr
D=load()
print(len(D))
for m,(a,b) in D.items():
    print(m.split('--')[1][:28],a['dataset'],a['n_docs'],a['n_queries'],round(a['baseline']['ndcg@10'],3),'|',b['n_docs'],b['n_queries'],round(b['baseline']['ndcg@10'],3),len(a['rows']),len(b['rows']))
