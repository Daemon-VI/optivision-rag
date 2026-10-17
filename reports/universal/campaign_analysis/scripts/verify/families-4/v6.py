exec(open(__file__.replace('v6.py','v.py')).read().split("for nm,only in")[0])
ds=DS[1]
for a,b in [('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-hf')]:
    pa={r['label']:r['retention'] for r in d[a][ds]['rows']};pb={r['label']:r['retention'] for r in d[b][ds]['rows']}
    diffs=sorted(((pa[l]-pb[l],l) for l in pa if l in pb),key=lambda t:-abs(t[0]))[:6]
    print(a,diffs)
    L=[l for l in pa if l in pb and 'project' not in l and l not in TRIV]
    print(' noproject RMSD %.4f n=%d'%(np.sqrt(np.mean([(pa[l]-pb[l])**2 for l in L])),len(L)))
