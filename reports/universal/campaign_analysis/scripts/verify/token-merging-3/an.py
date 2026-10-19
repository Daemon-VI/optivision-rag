import json,glob,os,numpy as np,collections
from scipy import stats
fs=[f.replace(os.sep,'/') for f in glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json')]
labs=['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)']
D={}; F16={}
for f in fs:
    d=json.load(open(f)); m=f.split('/campaign/')[1].split('/')[0]
    ds='docvqa' if 'docvqa' in f else 'infovqa' if 'infovqa' in f else 'scifact'
    F16[(ds,m)]='float16' in str(d['baseline']['encoder'].get('dtype'))
    for r in d['rows']:
        if r['label'] in labs:
            lo,hi=r['retention_ci:ndcg@5']
            D.setdefault((ds,r['label']),{})[m]=(1-r['retention:ndcg@5'],(hi-lo)/3.92)
print(collections.Counter(k[0] for k in D))
for ds in ['docvqa','infovqa','scifact']:
    for l in labs:
        x=D.get((ds,l),{})
        if not x: continue
        L=np.array([v[0] for v in x.values()]);S=np.array([v[1] for v in x.values()])
        print(ds,l,len(L),'mean loss %.2f sd %.2f meanSE %.2f share %.0f%%'%(L.mean()*100,L.std(ddof=1)*100,S.mean()*100,100*np.mean(S**2)/L.var(ddof=1)))
rng=np.random.default_rng(0)
def boot(x,y):
    bs=[]
    for _ in range(4000):
        i=rng.integers(0,len(x),len(x));bs.append(stats.spearmanr(x[i],y[i])[0])
    return np.nanpercentile(bs,[2.5,97.5]).round(2)
for l in labs:
    a=D[('docvqa',l)];b=D[('infovqa',l)];sh=sorted(set(a)&set(b))
    x=np.array([a[m][0] for m in sh]);y=np.array([b[m][0] for m in sh])
    print(l,len(sh),'rho %.2f'%stats.spearmanr(x,y)[0],boot(x,y),'p',round(stats.spearmanr(x,y)[1],3))
print('reliability (1 - meanSE^2/var) and ceiling on rho')
for l in labs:
    a=D[('docvqa',l)];b=D[('infovqa',l)];sh=sorted(set(a)&set(b))
    ra=1-np.mean([a[m][1]**2 for m in sh])/np.var([a[m][0] for m in sh],ddof=1)
    rb=1-np.mean([b[m][1]**2 for m in sh])/np.var([b[m][0] for m in sh],ddof=1)
    print(l,'doc %.2f info %.2f ceiling %.2f'%(ra,rb,np.sqrt(max(ra,0)*max(rb,0))))
# sibling robustness: drop models sharing name stem; and drop extreme model
print('robustness at keep 0.1 and 0.25')
for l in labs[2:4]:
    a=D[('docvqa',l)];b=D[('infovqa',l)];sh=sorted(set(a)&set(b))
    x=np.array([a[m][0] for m in sh]);y=np.array([b[m][0] for m in sh])
    k=np.argsort(-np.abs(x-x.mean()))[:3]
    keep=np.ones(len(sh),bool);keep[k]=False
    print(l,'drop 3 extreme-doc',round(stats.spearmanr(x[keep],y[keep])[0],2),'pearson',round(stats.pearsonr(x,y)[0],2))
    n16=np.array([F16[('docvqa',m)] for m in sh])
    print(' n f16',n16.sum(),'rho non-f16',round(stats.spearmanr(x[~n16],y[~n16])[0],2),'f16 vs rest MWU p',round(stats.mannwhitneyu(x[n16],x[~n16])[1],2) if n16.sum()>1 else None)
    # one per family stem (first token of name before '-' or '_')
    seen={};idx=[]
    for i,m in enumerate(sh):
        s=m.split('--')[1].lower().replace('-','').replace('_','')[:8]
        if s not in seen: seen[s]=1;idx.append(i)
    print(' stems',len(idx),'rho',round(stats.spearmanr(x[idx],y[idx])[0],2))
# SE noise sanity: SE ~ same across models?
L=D[('docvqa','hierarchical_merge(0.1)')]
print('SE range docvqa .1',min(v[1] for v in L.values())*100,max(v[1] for v in L.values())*100)
