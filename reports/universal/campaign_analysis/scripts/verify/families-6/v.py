import json,glob,numpy as np,os
from scipy.stats import spearmanr
from optivision.adapters import MODELS
d=json.load(open('src/optivision/presets/measured.json'))['models']
rng=np.random.default_rng(0)
out={}
for ds in ['vidore/docvqa_test_subsampled','vidore/infovqa_test_subsampled']:
    rows=[]
    for m,v in d.items():
        if ds not in v: continue
        r=[x for x in v[ds]['rows'] if x['label']=='hierarchical_merge(0.1)']
        if not r: continue
        info=MODELS.get(m) if hasattr(MODELS,'get') else None
        # base vectors from hm(0.1)? use rows with label float32/ none; find any row with no merge
        base=[x for x in v[ds]['rows'] if 'merge' not in x['label'] and 'token' not in x['label'] and x['vectors_per_doc']]
        vp=max(x['vectors_per_doc'] for x in base) if base else None
        rows.append((m,vp,r[0]['retention'],r[0]['vectors_per_doc']))
    out[ds]=rows
    print(ds,len(rows))
    a=np.array([(x[1],x[2],x[3]) for x in rows],float)
    for nm,mask in [('all',np.ones(len(a),bool)),('noColNetra',np.array([ 'ColNetra' not in x[0] for x in rows]))]:
        lv=np.log10(a[mask,0]);loss=1-a[mask,1]
        print(nm,mask.sum(),spearmanr(loss,lv))
    print('hm01 kept vs base ratio',np.round(a[:,2]/a[:,0],3).min(),np.round(a[:,2]/a[:,0],3).max())
    for x in sorted(rows,key=lambda x:x[1]):print(x)
print('--- processor-group means, and exclusions')
from collections import defaultdict
for ds,rows in out.items():
    for excl in [False,True]:
        g=defaultdict(list)
        for m,vp,ret,_ in rows:
            if excl and 'ColNetra' in m: continue
            g[round(vp,-1)].append(1-ret)
        x=np.log10(list(g.keys()));y=[np.mean(v) for v in g.values()]
        print(ds[-22:],'excl' if excl else 'all',len(x),spearmanr(x,y)[:2])
    # drop ColNetra + NeoMME + colsmolvlm (extreme vp)
    r2=[r for r in rows if not any(k in r[0] for k in ['ColNetra','NeoMME','colsmolvlm','colSmol'])]
    print('drop extremes',len(r2),spearmanr([1-r[2] for r in r2],[np.log10(r[1]) for r in r2])[:2])
    # permutation-bootstrap CI of rho excluding ColNetra, model bootstrap
    r3=[r for r in rows if 'ColNetra' not in r[0]]
    a=np.array([(np.log10(r[1]),1-r[2]) for r in r3]);bs=[]
    for _ in range(4000):
        i=rng.integers(0,len(a),len(a));bs.append(spearmanr(a[i,0],a[i,1])[0])
    print('boot CI',np.nanpercentile(bs,[2.5,97.5]))
