import json,glob,os,numpy as np
from scipy import stats
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/"
recs=[]
files=glob.glob(R+"campaign/*/wide/*.json")+glob.glob(R+"wide/*.json")
for f in files:
    b=os.path.basename(f)
    if b.startswith("smoke") or "ci" in b and False: continue
    try: d=json.load(open(f))
    except: continue
    if not isinstance(d,dict) or 'rows' not in d: continue
    rt={r['label']:r['retention:ndcg@5'] for r in d['rows']}
    model=f.replace("\\","/").split("/")
    model=model[-3] if "campaign" in f else b
    ds=d['dataset']
    for m in ['0.33','0.25']:
        mo=rt.get(f'hierarchical_merge({m})'); mi=rt.get(f'hierarchical_merge({m}) > int4'); mc=rt.get(f'hierarchical_merge({m}) > int4(mean)')
        recs.append(dict(model=model,ds=ds,m=m,mo=mo,mi=mi,mc=mc,i4=rt.get('int4'),i4c=rt.get('int4(mean)'),file=b))
print(len(files),len(recs))
import collections
text=lambda r:'scifact' in r['ds'].lower()
for name,sel in [('text',text),('image',lambda r:not text(r))]:
    rs=[r for r in recs if sel(r) and None not in (r['mo'],r['mi'],r['i4'])]
    print(name,len(rs),len({r['model'] for r in rs}))
    a=np.array([r['mi']/r['mo'] for r in rs]); b=np.array([r['i4'] for r in rs])
    print(' med merged-int4/merge',np.median(a),'int4 alone',np.median(b),'wilcoxon',stats.wilcoxon(a,b).pvalue,'frac>1pt extra',np.mean(b-a>0.01))
    # per-model-collapsed
    mm=collections.defaultdict(list)
    for r,x,y in zip(rs,a,b): mm[r['model']].append((x,y))
    am=np.array([np.mean([p[0] for p in v]) for v in mm.values()]);bm=np.array([np.mean([p[1] for p in v]) for v in mm.values()])
    print(' per-model n',len(am),'med',np.median(am),np.median(bm),'p',stats.wilcoxon(am,bm).pvalue, 'diff median',np.median(am-bm))
    # leave out LFM
    ex=[i for i,k in enumerate(mm) if 'LFM' not in k]
    print(' excl LFM p',stats.wilcoxon(am[ex],bm[ex]).pvalue,np.median(am[ex]-bm[ex]))
    # sign by model
    print(' models worse:',int((am<bm).sum()),'better',int((am>bm).sum()))
    # 0.25 vs 0.33
    d={}
    for r in rs: d.setdefault(r['model']+r['ds'],{})[r['m']]=r['mi']/r['mo']
    print(' 0.25<0.33',sum(v['0.25']<v['0.33'] for v in d.values() if len(v)==2),'>',sum(v['0.25']>v['0.33'] for v in d.values() if len(v)==2))
    # bootstrap CI median diff (resample models)
    rng=np.random.default_rng(0);dd=am-bm
    bs=[np.median(rng.choice(dd,len(dd))) for _ in range(5000)];print(' boot CI median diff',np.percentile(bs,[2.5,97.5]))
    print(' per-model diff sorted:',sorted([(round(x-y,3),k[:25]) for (k,_),x,y in zip(mm.items(),am,bm)])[:6])
    # centred
    rc=[r for r in recs if sel(r) and None not in (r['mo'],r['mc'])]
    c=np.array([r['mc']/r['mo'] for r in rc]);print(' centred rows',len(rc),'<0.97:',[(r['model'][:20],round(r['mc']/r['mo'],3)) for r in rc if r['mc']/r['mo']<0.97])
    print(' mo range',min(r['mo'] for r in rs),np.median([r['mo'] for r in rs]))
# LFM
for r in recs:
    if 'LFM2-Col' in r['model']: print(r)
