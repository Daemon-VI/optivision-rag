import os,json,glob,numpy as np,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
rs=[];dss=collections.Counter()
for f in sorted(glob.glob(R+'*/wide/*.json')):
    d=json.load(open(f)); m=os.path.basename(os.path.dirname(os.path.dirname(f))); dss[d['dataset']]+=1
    ret={r['label']:r['retention:ndcg@5'] for r in d['rows']}
    for l,v in ret.items():
        P=l.split(' > ')
        if len(P)<2 or 'float16' in P: continue
        # any split into measured prefix + measured suffix, or full singles
        cands=[]
        if all(p in ret for p in P): cands.append([ret[p] for p in P])
        for i in range(1,len(P)):
            a,b=' > '.join(P[:i]),' > '.join(P[i:])
            if a in ret and b in ret: cands.append([ret[a],ret[b]])
        if cands: rs.append((m,d['dataset'],l,v,cands[0],len(P)))
print(dss)
a=np.array([r[3] for r in rs]);p=np.array([np.prod(r[4]) for r in rs]);ad=np.array([1-sum(1-x for x in r[4]) for r in rs])
print(len(rs),len({r[0] for r in rs}),'prod MAE',abs(p-a).mean(),'add',abs(ad-a).mean(),'bias',(p-a).mean(),'max',(p-a).max(),'min',(p-a).min())
# paired diff prod vs additive by bootstrap over models
ms=sorted({r[0] for r in rs}); idx=collections.defaultdict(list)
for i,r in enumerate(rs): idx[r[0]].append(i)
dd=abs(p-a)-abs(ad-a); rng=np.random.default_rng(1)
bs=[np.concatenate([dd[idx[ms[j]]] for j in rng.integers(0,len(ms),len(ms))]).mean() for _ in range(2000)]
print('MAE prod-add diff',dd.mean(),np.percentile(bs,[2.5,97.5]))
# sibling dedupe: keep one per family prefix (org)
org=collections.defaultdict(list)
for m in ms: org[m.split('--')[0]].append(m)
keep={v[0] for v in org.values()}
k=[i for i,r in enumerate(rs) if r[0] in keep]; print('one per org',len(keep),len(k),'MAE',abs(p-a)[k].mean(),'add',abs(ad-a)[k].mean())
tx=[i for i,r in enumerate(rs) if 'scifact' in r[1].lower()]; print('text rows',len(tx), abs(p-a)[tx].mean() if tx else '')
