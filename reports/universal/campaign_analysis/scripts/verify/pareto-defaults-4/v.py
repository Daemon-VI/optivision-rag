import json,glob,numpy as np,os
from scipy import stats
os.chdir('C:/Users/Lenovo/projects/optivision-package/reports/universal')
fs=sorted(glob.glob('campaign/*/wide/*_*.json')+glob.glob('wide/*.json'))
R=[]
for f in fs:
    d=json.load(open(f))
    if not d.get('tiered'): continue
    rows={r['label']:r for r in d['rows']}
    m='retention:ndcg@5';ci='retention_ci:ndcg@5'
    T={t['hot']:t for t in d['tiered']}
    i8=[r for r in d['rows'] if r['label'] in('int8(per_vector)',)]
    R.append(dict(f=f,ds=d['dataset'],img='scifact' not in d['dataset'].lower(),nd=d['n_docs'],
      tb=T['binary'][m],tbl=T['binary'][ci][0],tw=T['ward 1/4 > binary'][m],twl=T['ward 1/4 > binary'][ci][0],
      ub=rows['binary'][m],ubl=rows['binary'][ci][0],uw=rows['hierarchical_merge(0.25) > binary'][m],uwl=rows['hierarchical_merge(0.25) > binary'][ci][0],
      i8=rows['int8(per_vector)'][m] if 'int8(per_vector)' in rows else np.nan, ram=T['binary']['ram_compression_vs_float32'],dk=T['binary']['disk_compression_vs_float32']))
print(len(R),sum(r['img'] for r in R),set(r['nd'] for r in R))
def cov(k,th,s=R): return sum(r[k]>=th for r in s)
for nm,s in [('all',R),('img',[r for r in R if r['img']]),('sci',[r for r in R if not r['img']])]:
    print(nm,len(s))
    for th in(.95,.97,.99):print(' ',th,'tiered',cov('tbl',th,s),'untiered',cov('ubl',th,s),'tw',cov('twl',th,s),'uw',cov('uwl',th,s))
def gain(a,b,s=R):
    x=np.array([r[a]-r[b] for r in s]);rng=np.random.default_rng(0)
    bs=[rng.choice(x,len(x)).mean() for _ in range(5000)]
    return x.mean(),np.percentile(bs,[2.5,97.5]),stats.wilcoxon(x).pvalue
print('gain bin',gain('tb','ub'));print('gain ward',gain('tw','uw'))
print('i8 nan',sum(np.isnan(r['i8']) for r in R))
x=np.array([r['tb']-r['i8'] for r in R if not np.isnan(r['i8'])]);print('tb-i8',x.mean(),x.min(),len(x))
print('ram/disk',sorted(set(round(r['dk'],1) for r in R))[:3],max(r['dk'] for r in R),min(r['dk'] for r in R))
for r in R:
    if r['tbl']<.95:print(r['f'].split('/')[-1],r['tb'],r['tbl'],r['ub'],r['nd'])
# sibling dedup: one per model (campaign dir / file prefix), and drop mxbai-edge
import collections
g=collections.defaultdict(list)
for r in R:g[r['f'].replace(chr(92),'/').split('/')[-1].split('_')[0]].append(r)
print('models',len(g))
D=[np.mean([r['tbl'] for r in v])>=0.95 for v in g.values()];print('model-level mean lb>=.95',sum(D),len(D))
S=[r for r in R if 'v0-17m' not in r['f']]
print('gain w/o 17m',gain('tb','ub',S)[:2])
print('median tb-ub text/img',np.median([r['tb']-r['ub'] for r in R if not r['img']]),np.median([r['tb']-r['ub'] for r in R if r['img']]))
