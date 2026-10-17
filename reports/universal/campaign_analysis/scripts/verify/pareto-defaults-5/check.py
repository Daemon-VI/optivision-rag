import json,glob,os,collections,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
files=sorted(glob.glob(R+'campaign/*/wide/*.json'))+sorted(glob.glob(R+'wide/*_*_test_subsampled.json'))
print(len(files))
pairs=[]
for f in files:
    f=f.replace(chr(92),'/'); d=json.load(open(f)); 
    model=os.path.basename(f).rsplit('_',2)[0] if 'campaign' not in f else f.split('campaign/')[1].split('/')[0]
    rows=[]
    for r in d['rows']:
        rows.append(dict(label=r['label'],c=r['compression_vs_float32'],ret=r['retention:ndcg@5'],lo=r['retention_ci:ndcg@5'][0],fl=bool(r.get('pareto'))))
    pairs.append(dict(f=f,model=model,ds=d['dataset'],rows=rows,base=d['baseline']))
print(len(set(p['model'] for p in pairs)), 'distinct model keys')
def front(rows,key):
    out=[]
    for i,r in enumerate(rows):
        dom=any((o['c']>=r['c'] and o[key]>=r[key]) and (o['c']>r['c'] or o[key]>r[key]) for o in rows)
        if not dom: out.append(r)
    return out
nf=[];nflag=[]
frows=[];flagrows=[]
for p in pairs:
    fr=front(p['rows'],'ret'); nf.append(len(fr)); frows+=fr
    fl=[r for r in p['rows'] if r['fl']]; nflag.append(len(fl)); flagrows+=fl
print('front pts/pair',np.mean(nf),'flag',np.mean(nflag))
print('flag rows ret<0.90',np.mean([r['ret']<0.90 for r in flagrows]),'lo<0.95',np.mean([r['lo']<0.95 for r in flagrows]), len(flagrows))
print('recomp front ret<0.90',np.mean([r['ret']<0.90 for r in frows]),'lo<.95',np.mean([r['lo']<0.95 for r in frows]))
cnt=collections.Counter(r['label'] for r in frows); print(cnt.most_common(8))
# restricted: lo>=.95, non-dominated on (c,lo)
cnt2=collections.Counter()
for p in pairs:
    u=[r for r in p['rows'] if r['lo']>=0.95]
    for r in front(u,'lo'): cnt2[r['label']]+=1
print(cnt2.most_common(6))
# oracle
rng=np.random.default_rng(0)
orc={}
for T in (.95,.97,.99):
    v=[];w=collections.Counter()
    for p in pairs:
        u=[r for r in p['rows'] if r['lo']>=T]
        b=max(u,key=lambda r:r['c']); v.append(b['c']); w[b['label']]+=1
    v=np.array(v); orc[T]=v
    print(T,'median',np.median(v),'IQR',np.percentile(v,[25,75]),w.most_common(3))
# blind default: single label with lo>=T on >=95% of pairs, max median? compute label coverage
labs=collections.defaultdict(list)
for p in pairs:
    for r in p['rows']: labs[r['label']].append(r)
for T in (.95,):
    res=[]
    for l,rs in labs.items():
        if len(rs)<80: continue
        cov=np.mean([r['lo']>=T for r in rs]); res.append((cov,np.median([r['c'] for r in rs]),np.mean([r['c'] for r in rs]),l,len(rs)))
    ok=[x for x in res if x[0]>=0.95]; ok.sort(key=lambda x:-x[1]); print('blind >=95% cov',ok[:4])
    ok=[x for x in res if x[0]>=0.90]; ok.sort(key=lambda x:-x[1]); print('blind >=90% cov',ok[:3])
# per-pair sensitivity: bootstrap over pairs for oracle median
v=orc[.95]
bs=[np.median(rng.choice(v,len(v))) for _ in range(5000)]; print('oracle med CI',np.percentile(bs,[2.5,97.5]))
# by dataset
for ds in set(p['ds'] for p in pairs): print(ds,[ (T,np.median([orc[T][i] for i,p in enumerate(pairs) if p['ds']==ds])) for T in orc])
# sibling/ drop dups : unique model keys
# float16 baseline pairs
fp=[i for i,p in enumerate(pairs) if 'float16' in json.dumps(p['base']).lower()]; print('f16 baseline pairs',len(fp))
print(json.dumps(pairs[0]['base'])[:300])
# selection
sel=[];
sf=sorted(glob.glob(R+'campaign/*/selection_rules/*.json'));print(len(sf))
agg=collections.defaultdict(lambda:[[],[],[]])
for f in sf:
    d=json.load(open(f))
    for e in d['results']:
        if e['reference']!='labels' or e['target']!=0.95: continue
        a=agg[e['rule']]; a[0].append(e['met']); a[1].append(e['compression_median']); a[2].append(e['met_labels'])
for k,(m,c,ml) in agg.items(): print(k,len(m),'met',np.mean(m),'all20',np.mean(np.array(m)>=1),'comp med',np.median(c),'labels met',np.mean(ml))
print(collections.Counter((e['reference'],e['target']) for e in d['results']))
