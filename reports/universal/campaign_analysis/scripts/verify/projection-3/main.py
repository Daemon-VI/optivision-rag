from load import *
from scipy import stats
rng=np.random.default_rng(1)
pairs=[]
for (m,ds),v in sorted(U.items()):
    d=v[0][1]; R_=rows(d); D=d['baseline']['dim']
    for l in R_:
        mm=re.fullmatch(r'project\((\d+), random\)',l)
        if mm:
            k=int(mm.group(1)); p=f'project({k})'
            if p in R_ and D>128:
                pairs.append((m,ds,D,k,R_[p][M],R_[l][M],R_[p]['retention_interval'] if 'retention_interval' in R_[p] else None))
            elif D>128: print('no pca partner',m,ds,D,k)
print(len(pairs))
for p in pairs: print(p[:6],round(p[4]-p[5],4))
models=sorted({p[0] for p in pairs}); print(len(models),models)
diff=np.array([p[4]-p[5] for p in pairs])*100
print('units mean',diff.mean(),'ahead',(diff>0).sum(),len(diff))
def agg(ps):
    by={}
    for p in ps: by.setdefault(p[0],[]).append((p[4]-p[5])*100)
    return by
by=agg(pairs); mm=np.array([np.mean(v) for v in by.values()])
print('model means',dict(zip(by,np.round(mm,2))))
print('mean of model means',mm.mean(),'ahead',(mm>0).sum(),len(mm))
bs=[]
keys=list(by)
for _ in range(20000):
    s=rng.choice(len(keys),len(keys)); bs.append(np.mean([ by[keys[i]] for i in s ] and np.concatenate([by[keys[i]] for i in s])))
print('cluster boot units-mean CI',np.percentile(bs,[2.5,97.5]))
bs=[mm[rng.integers(0,len(mm),len(mm))].mean() for _ in range(20000)]
print('model-mean boot CI',np.percentile(bs,[2.5,97.5]))
print('sign units p',stats.binomtest(int((diff>0).sum()),int((diff!=0).sum())).pvalue)
print('sign models p',stats.binomtest(int((mm>0).sum()),len(mm)).pvalue)
print('wilcoxon models',stats.wilcoxon(mm).pvalue,'units',stats.wilcoxon(diff).pvalue)
# by D
for D in sorted({p[2] for p in pairs}):
    x=[(p[4]-p[5])*100 for p in pairs if p[2]==D]; print('D',D,len(x),np.mean(x))
# by dataset
for ds in ['docvqa','infovqa']:
    x=[(p[4]-p[5])*100 for p in pairs if p[1]==ds]; print(ds,len(x),np.mean(x))
# leave-one-model-out
for m in keys:
    r=np.mean([x for k2 in keys if k2!=m for x in by[k2]]); print('LOO',m[:40],round(r,2))
# leave out nemotron
big=[k for k in keys if 'emotron' in k or 'EVIE' in k]
print('no 2048+:',np.mean([x for k in keys if k not in big for x in by[k]]), 'models',len(keys)-len(big))
# sibling families
fam=lambda m: re.sub(r'[-_.0-9]+(b|B)$','',m.split('--')[-1]).split('-')[0].lower()
fm={}
for k in keys: fm.setdefault(fam(k),[]).extend(by[k])
print({k:(len(v),round(np.mean(v),2)) for k,v in fm.items()})
fmm=np.array([np.mean(v) for v in fm.values()]); print('family means',fmm.mean(),(fmm>0).sum(),len(fmm),stats.binomtest(int((fmm>0).sum()),len(fmm)).pvalue)
# pair table details
for p in pairs:
    if p[3] in(640,512,1024): print(p[:6])
