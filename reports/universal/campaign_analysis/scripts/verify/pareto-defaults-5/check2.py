import json,glob,os,collections,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
files=[f.replace(chr(92),'/') for f in sorted(glob.glob(R+'campaign/*/wide/*.json'))+sorted(glob.glob(R+'wide/*_*_test_subsampled.json'))]
P=[]
for f in files:
    d=json.load(open(f)); m=f.split('campaign/')[1].split('/')[0] if 'campaign' in f else os.path.basename(f).rsplit('_',3)[0]
    P.append(dict(model=m,ds=d['dataset'],rows=[(r['label'],r['compression_vs_float32'],r['retention:ndcg@5'],r['retention_ci:ndcg@5'][0]) for r in d['rows']]))
N=len(P);print(N)
def blind(idx,T,use):  # use 2=point 3=lo
    labs=collections.defaultdict(list)
    for i in idx:
        for r in P[i]['rows']: labs[r[0]].append((r[1],r[use]>=T))
    best=None
    for l,v in labs.items():
        if len(v)<len(idx): continue
        cov=np.mean([x[1] for x in v])
        if cov>=0.95:
            c=np.median([x[0] for x in v])
            if best is None or c>best[0]: best=(c,l,cov)
    return best
def orc(idx,T,use):
    return np.array([max(r[1] for r in P[i]['rows'] if r[use]>=T) for i in idx])
allidx=list(range(N))
for T in (.95,.97,.99):
    for use,nm in ((3,'lo'),(2,'pt')):
        b=blind(allidx,T,use); o=orc(allidx,T,use)
        print(T,nm,'blind',b,'oracle med',np.median(o),'ratio',np.median(o)/b[0] if b else None)
# oracle restricted to 59 selection files' pairs
sf=glob.glob(R+'campaign/*/selection_rules/*.json'); keys=set()
for f in sf:
    f=f.replace(chr(92),'/'); d=json.load(open(f)); keys.add((f.split('campaign/')[1].split('/')[0],d['dataset']))
idx59=[i for i,p in enumerate(P) if (p['model'],p['ds']) in keys]; print('matched',len(idx59))
for use in (3,2):
    o=orc(idx59,.95,use); b=blind(idx59,.95,use); print('59 pairs oracle',np.median(o),'blind',b)
# subgroups
for name,sel in (('docvqa',lambda p:'docvqa' in p['ds']),('infovqa',lambda p:'infovqa' in p['ds']),('scifact',lambda p:'scifact' in p['ds'])):
    idx=[i for i,p in enumerate(P) if sel(p)]; o=orc(idx,.95,3); print(name,len(idx),np.median(o),np.percentile(o,[25,75]),blind(idx,.95,3))
# one dataset per model (docvqa-first) to avoid pair duplicates, and drop models with 'colpali' siblings
seen=set();idx=[]
for i,p in enumerate(P):
    if p['model'] not in seen: seen.add(p['model']); idx.append(i)
o=orc(idx,.95,3);print('one pair/model',len(idx),np.median(o),blind(idx,.95,3))
# bootstrap by model for ratio
rng=np.random.default_rng(1)
models=sorted(set(p['model'] for p in P)); bym=collections.defaultdict(list)
for i,p in enumerate(P): bym[p['model']].append(i)
rat=[]
for _ in range(1000):
    ms=rng.choice(len(models),len(models)); idx=[i for k in ms for i in bym[models[k]]]
    o=orc(idx,.95,3); 
    rat.append(np.median(o))
print('cluster-boot oracle median CI',np.percentile(rat,[2.5,97.5]))
# calibrate default vs oracle on 59, per-file
cal=[]
for f in sf:
    f=f.replace(chr(92),'/'); d=json.load(open(f))
    for e in d['results']:
        if e['reference']=='labels' and e['target']==0.95 and e['rule']=='calibrate() default': cal.append((f.split('campaign/')[1].split('/')[0],d['dataset'],e['compression_median'],e['met']))
print('calibrate median',np.median([c[2] for c in cal]))
bm=collections.defaultdict(list)
for c in cal: bm[c[0]].append(c[2])
print('calibrate model-level median',np.median([np.mean(v) for v in bm.values()]), 'met by dataset',{k:np.mean([c[3] for c in cal if k in c[1]]) for k in ('docvqa','infovqa','scifact')})
print('dataset counts',collections.Counter(c[1] for c in cal))
