from load import *
for k,v in recs.items():
    for f,d in v:
        L=[r['label'] for r in d['rows'] if 'random' in r['label'] and r['group'] in('dim',)]
        if L and d['baseline']['dim']>128: print(k,d['baseline']['dim'],L)
