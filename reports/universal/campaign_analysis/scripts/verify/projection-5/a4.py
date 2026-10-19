import os,json,glob,collections,re,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
data=collections.defaultdict(dict)
for f in glob.glob(R+'*/wide/*.json'):
    j=json.load(open(f)); mod=os.path.normpath(f).replace(os.sep,'/').split('/campaign/')[1].split('/')[0]
    ds='doc' if 'docvqa' in j['dataset'].lower() else 'info' if 'infovqa' in j['dataset'].lower() else None
    if not ds: continue
    data[mod][ds]=dict(rows={r['label']:r['retention:ndcg@5'] for r in j['rows']},base=j['baseline']['ndcg@5'],dim=j['baseline']['dim'])
# also older measured.json models
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
for mod,d in m.items():
    key=mod.replace('/','--')
    if key in data or D not in d or I not in d: continue
    data[key]={'doc':dict(rows={r['label']:r['retention'] for r in d[D]['rows']},base=d[D]['baseline'],dim=None),'info':dict(rows={r['label']:r['retention'] for r in d[I]['rows']},base=d[I]['baseline'],dim=None)}
mods=[x for x in data if 'doc' in data[x] and 'info' in data[x]]
print(len(data),len(mods))
labs=collections.Counter()
for x in mods:
    for l in set(data[x]['doc']['rows'])&set(data[x]['info']['rows']): labs[l]+=1
rng=np.random.default_rng(2)
out=[]
for l,n in labs.items():
    mk=re.findall(r'project\((\d+)',l)
    if not mk or n<5: continue
    k=int(mk[-1])
    if k>40 and 'merge' not in l: continue
    d=np.array([data[x]['info']['rows'][l]-data[x]['doc']['rows'][l] for x in mods if l in data[x]['doc']['rows'] and l in data[x]['info']['rows']])
    bs=[rng.choice(d,len(d)).mean() for _ in range(3000)]
    p=stats.wilcoxon(d).pvalue if len(d)>5 else np.nan
    out.append((k,l,n,d.mean()*100,np.percentile(bs,2.5)*100,np.percentile(bs,97.5)*100,(d>0).mean(),p))
for o in sorted(out): print("k=%d %-60s n=%d diff=%.2f [%.2f,%.2f] frac>0=%.2f p=%.2g"%o)
print('=====extra')
def get(l):
    r=[(x,data[x]['doc']['rows'][l],data[x]['info']['rows'][l]) for x in mods if l in data[x]['doc']['rows'] and l in data[x]['info']['rows']]
    return r
for l in ['hierarchical_merge(0.33) > project(16)','hierarchical_merge(0.33) > project(32)','project(32, random)','hierarchical_merge(0.33) > project(128)']:
    r=get(l); d=np.array([b-a for _,a,b in r]); da=np.array([a for _,a,b in r]); ia=np.array([b for _,a,b in r])
    print(l,'n',len(r),'doc mean %.3f info mean %.3f'%(da.mean(),ia.mean()),'median diff %.2f'%(np.median(d)*100))
    # remove top 3 extreme
    s=np.sort(d); print('  trimmed(3 each) %.2f'%(s[3:-3].mean()*100))
    # family-level
    fam=collections.defaultdict(list)
    for (x,a,b) in r: fam[x.split('--')[0]].append(b-a)
    fv=np.array([np.mean(v) for v in fam.values()]); print('  families',len(fv),'mean %.2f frac>0 %.2f p=%.2g'%(fv.mean()*100,(fv>0).mean(),stats.wilcoxon(fv).pvalue))
    # leave-one-family-out min
    lo=[np.mean([np.mean(v) for f,v in fam.items() if f!=g]) for g in fam]; print('  LOFO range %.2f..%.2f'%(min(lo)*100,max(lo)*100))
    # info/doc loss ratio
    print('  loss ratio median %.2f'%np.median((1-ia)/(1-da)))
    # relation to doc baseline
    bb=[data[x]['doc']['base'] for x,_,_ in r]; print('  spearman gap~docbase',stats.spearmanr(d,bb)[0].round(2),'gap~infobase',stats.spearmanr(d,[data[x]['info']['base'] for x,_,_ in r])[0].round(2))
    # float16
    f16=[('float16' in (m.get(x.replace('--','/'),{}).get(D,{}).get('note',''))) for x,_,_ in r]; f16=np.array(f16)
    if f16.any() and (~f16).any(): print('  f16 n=%d gap %.2f; f32 gap %.2f'%(f16.sum(),d[f16].mean()*100,d[~f16].mean()*100))
# baseline correlation 128d models, k=64
r=get('project(64)'); 
