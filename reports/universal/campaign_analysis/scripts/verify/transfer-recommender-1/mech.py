import sys
sys.argv = ['x', '10']
from an2 import *
for kk in [k for k in keys if k.startswith('hierarchical_merge') and 'pca0.25' in k and k.count('>') == 1 and 'binary' not in k and 'int' not in k]:
    j = keys.index(kk)
    print(kk)
    for b in ['<=96', '128', '320-640', '>=2048']:
        for mo in ['image', 'text']:
            s = (bd == b) & (mod == mo) & V[:, j]
            if s.sum(): print('  ', b, mo, 'units', s.sum(), 'pass.95', int((Y[s, j] >= .95).sum()), 'mean ret', round(float(Y[s, j].mean()), 3))
