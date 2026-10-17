exec(open('a.py').read().split("for t in (.95,.97,.99):\n    print(t,summ(run(t),t))")[0])
import collections
J=(.5,.5)
for t in (.95,.97,.99):
    res=run(t,prior=J)
    sel=[r for r in res if r[1] is not None]
    # by bucket
    um={}
    for u in units: um.setdefault(u['model'],u)
    bk=collections.Counter()
    for u,r in zip([u for u in units if True],[None]*0): pass
    # recompute bucket stats
    stat=collections.defaultdict(lambda:[0,0,0])
    i=0
    for u in units:
        peers=[p for p in units if p['model']!=u['model'] and p['dim']==u['dim'] and p['mod']==u['mod']]
        if not peers: stat[(u['dim'],u['mod'])][2]+=1;continue
    # selection by bucket
    allres=[]
    for u in units:
        r=run.__wrapped__ if False else None
    print(t,'selected',len(sel),'of',len(res))
# bucket-level: rerun with unit info
def run2(t,prior=J,q=.1,c=.9,drop=lambda u:False):
    out=[]
    for u in units:
        if drop(u): continue
        peers=[p for p in units if p['model']!=u['model'] and p['dim']==u['dim'] and p['mod']==u['mod'] and not drop(p)]
        best=None
        for lab,r in u['rows'].items():
            ps=[p['rows'][lab] for p in peers if lab in p['rows']]
            if not ps: continue
            k=sum(x['retention']>=t for x in ps);n=len(ps)
            if beta.ppf(q,prior[0]+k,prior[1]+n-k)>=c:
                comp=np.median([x['compression'] for x in ps])
                if best is None or comp>best[0]:best=(comp,lab)
        out.append((u,len(peers),best and u['rows'][best[1]]))
    return out
for t in (.95,.97,.99):
    o=run2(t)
    b=collections.defaultdict(lambda:[0,0,0,0])
    for u,np_,r in o:
        k=(u['dim'],u['mod']);b[k][0]+=1
        if r: b[k][1]+=1;b[k][2]+=r['retention']>=t
    print(t,{k:v[:3] for k,v in b.items()})
# robustness: image-128 only, drop sibling families, drop most extreme
def fam(m): return info[m].family
for t in (.95,.97,.99):
    for name,dr in [('text',lambda u:u['mod']!='text'),('image128',lambda u:not(u['mod']=='image' and u['dim']==128))]:
        o=[x for x in run2(t,drop=dr) if x[2]]
        h=[x[2]['retention']>=t for x in o];l=[x[2]['retention_interval'][0]>=t for x in o]
        print(t,name,len(o),np.mean(h) if h else None,np.mean(l) if l else None,np.median([x[2]['compression'] for x in o]) if o else None)
# family-level LOFO (exclude same family)
def run3(t,prior=J):
    o=[]
    for u in units:
        peers=[p for p in units if fam(p['model'])!=fam(u['model']) and p['dim']==u['dim'] and p['mod']==u['mod']]
        best=None
        for lab,r in u['rows'].items():
            ps=[p['rows'][lab] for p in peers if lab in p['rows']]
            if not ps: continue
            k=sum(x['retention']>=t for x in ps);n=len(ps)
            if beta.ppf(.1,prior[0]+k,prior[1]+n-k)>=.9:
                comp=np.median([x['compression'] for x in ps])
                if best is None or comp>best[0]:best=(comp,lab)
        if best:o.append(u['rows'][best[1]])
    return o
print(collections.Counter(fam(u['model']) for u in units).most_common(12))
for t in (.95,.97,.99):
    o=run3(t);print('LOFO',t,len(o),np.mean([r['retention']>=t for r in o]),np.mean([r['retention_interval'][0]>=t for r in o]),np.median([r['compression'] for r in o]))
