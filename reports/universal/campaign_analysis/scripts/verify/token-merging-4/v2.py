import json,re,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
labs=['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)']
rng=np.random.default_rng(1)
dn=['vidore/docvqa_test_subsampled','vidore/infovqa_test_subsampled']
def g(m,ds,l):
    for r in d[m][ds]['rows']:
        if r['label']==l: return r['retention']
ms=[m for m,v in d.items() if len(v)>=2 and g(m,dn[0],labs[0]) is not None]
print(len(ms))
f16=[m for m in ms if any('float16' in str(d[m][ds]['rows'][0])+str(d[m][ds].get('note','')) for ds in dn)]
print('f16-note models',len(f16))
def fam(m):
    s=m.split('/')[-1].lower()
    s=re.sub(r'(-hf|-merged|-v?\d[\d.]*$)','',s); return s[:8]
def run(S,tag):
    print('==',tag,len(S))
    for l in labs:
        a=np.array([1-g(m,dn[0],l) for m in S]);b=np.array([1-g(m,dn[1],l) for m in S]);df=a-b
        bs=[rng.choice(df,len(df)).mean() for _ in range(4000)]
        print(l,'ratio %.2f medratio %.2f'%(a.mean()/b.mean(),np.median(a)/np.median(b)),'diff %.2f [%.2f,%.2f] W p=%.3f docworse %d/%d'%(df.mean()*100,*np.percentile(bs,[2.5,97.5])*np.array([100,100]),stats.wilcoxon(df).pvalue,(df>0).sum(),len(df)))
run(ms,'all')
run([m for m in ms if m not in f16],'nof16')
run([m for m in ms if 'ColNetra' not in m],'drop ColNetra')
seen={};
for m in ms: seen.setdefault(fam(m),m)
print(len(seen),'families');run(list(seen.values()),'one per family')
# baselines & spearman for adaptive
B=lambda m,ds:d[m][ds]['baseline']
bd=np.array([B(m,dn[0]) for m in ms]);bi=np.array([B(m,dn[1]) for m in ms]);print('baseline mean',bd.mean(),bi.mean())
# relative-to-headroom loss
for l in labs:
    a=np.array([1-g(m,dn[0],l) for m in ms]);b=np.array([1-g(m,dn[1],l) for m in ms])
    print(l,'abs ndcg loss doc %.4f info %.4f'%((a*bd).mean(),(b*bi).mean()))
    print(' spearman doc',stats.spearmanr(bd,a)[:2],'info',stats.spearmanr(bi,b)[:2])
