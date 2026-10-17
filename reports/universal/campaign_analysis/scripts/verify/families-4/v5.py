exec(open(__file__.replace('v5.py','v.py')).read().split("for nm,only in")[0])
exec(open(__file__.replace('v5.py','v3.py')).read().split("for ds in DS:")[0].split("img=")[0].split("exec(")[0]) if False else None
def rmsd(a,b,ds,flt=lambda l:True):
    pa={r['label']:r['retention'] for r in d[a][ds]['rows']};pb={r['label']:r['retention'] for r in d[b][ds]['rows']}
    L=[l for l in pa if l in pb and l not in TRIV and flt(l)]
    return np.sqrt(np.mean([(pa[l]-pb[l])**2 for l in L])),len(L), np.mean([(pa[l]-pb[l]) for l in L])
tw=[('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),('vidore/colqwen2-v1.0-merged','vidore/colqwen2-v1.0-hf')]
rep=[('vidore/colpali-v1.2','vidore/colpali-v1.2-merged'),('vidore/colpali-v1.3','vidore/colpali-v1.3-merged'),('tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged')]
for nm,P in [('twin',tw),('replicate',rep)]:
  for a,b in P:
    for ds in DS:
        if ds in d[a] and ds in d[b]:
            r=rmsd(a,b,ds); print(nm,a.split('/')[1],b.split('/')[1][-12:],ds[6:12],'RMSD %.4f n=%d meandiff(a-b) %+.4f'%r,'baselines',d[a][ds]['baseline'],d[b][ds]['baseline'])
# bootstrap-free check on label noise: twins with query-level? just per-group
for a,b in tw:
    ds=DS[0]
    for nm,f in [('codec',lambda l:'merge' not in l and 'project' not in l),('merge',lambda l:'merge' in l and 'project' not in l and '>' not in l),('rest',lambda l:'project' in l or '>' in l)]:
        r=rmsd(a,b,ds,f); print(a.split('/')[1],nm,'RMSD %.4f n=%d meandiff %+.4f'%r)
# Spearman leave-one-out infovqa/docvqa
size=eval(open(__file__.replace('v5.py','v3.py')).read().split("size=")[1].split("\nfor ds")[0])
img=[k for k in d if reg[k].modality=='image' and 'binary' in prof(k,DS[0])]
for ds in DS:
    ks=[k for k in size if k in img]
    L=[l for l in prof(ks[0],ds) if 'project' not in l and l not in TRIV]
    y={k:np.mean([prof(k,ds)[l] for l in L]) for k in ks}
    x={k:np.log(size[k]) for k in ks}
    rho=stats.spearmanr([x[k] for k in ks],[y[k] for k in ks]).statistic
    loo=[stats.spearmanr([x[j] for j in ks if j!=k],[y[j] for j in ks if j!=k]).statistic for k in ks]
    print(ds[6:12],len(ks),'rho %.2f LOO range %.2f..%.2f'%(rho,min(loo),max(loo)))
    # one per size-per-family: family-mean
    import collections
    fam=collections.defaultdict(list)
    for k in ks: fam[reg[k].family].append(k)
    # within-family demeaned
    xd=[];yd=[]
    for f,v in fam.items():
        if len(v)>1:
            mx=np.mean([x[k] for k in v]);my=np.mean([y[k] for k in v])
            for k in v: xd.append(x[k]-mx);yd.append(y[k]-my)
    print('  within-family-demeaned spearman',stats.spearmanr(xd,yd), 'n',len(xd))
