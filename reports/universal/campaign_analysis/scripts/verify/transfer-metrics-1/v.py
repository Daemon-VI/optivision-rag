import json,numpy as np,re
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
P=[]
for m,v in d.items():
    if D in v and I in v:
        a={r['label']:r for r in v[D]['rows']};b={r['label']:r for r in v[I]['rows']}
        for l in a:
            if l in b and l!='float32': P.append((m,l,a[l],b[l],v[D]['baseline'],v[I]['baseline']))
print('pairs',len(P),'models',len({p[0] for p in P}))
rd=np.array([p[2]['retention'] for p in P]);ri=np.array([p[3]['retention'] for p in P])
df=ri-rd
print('mean',df.mean(),'median',np.median(df),'frac>0',(df>0).mean())
ms=sorted({p[0] for p in P})
mm=np.array([np.median([x for x,p in zip(df,P) if p[0]==m]) for m in ms])
print('pos',(mm>0).sum(),len(ms),stats.binomtest(int((mm>0).sum()),len(ms)).pvalue)
mn=np.array([df[[p[0]==m for p in P]].mean() for m in ms]);rng=np.random.default_rng(0)
bs=[rng.choice(mn,len(mn)).mean() for _ in range(5000)];print('modelmean',mn.mean(),np.percentile(bs,[2.5,97.5]))
lo_d=np.array([p[2]['retention_interval'][0] for p in P]);lo_i=np.array([p[3]['retention_interval'][0] for p in P])
sd=lo_d>=.95;si=lo_i>=.95
print('safe',sd.sum(),si.sum(),'both',(sd&si).sum(),'D only',(sd&~si).sum(),'I only',(~sd&si).sum(),'P(I|D)',(sd&si).sum()/sd.sum(),'P(D|I)',(sd&si).sum()/si.sum())
print('point>=.95',((rd>=.95)&(ri<.95)).sum(),((rd<.95)&(ri>=.95)).sum())
# widths
wd=np.array([p[2]['retention_interval'][1]-p[2]['retention_interval'][0] for p in P]);wi=np.array([p[3]['retention_interval'][1]-p[3]['retention_interval'][0] for p in P])
print('width med',np.median(wd),np.median(wi))
# width-matched: lower = point - halfwidth_D for both
h=wd/2;print('wm',((rd-h>=.95)&(ri-h<.95)).sum(),((rd-h<.95)&(ri-h>=.95)).sum())
# loss ratio, symmetric selection mean<0.97
mask=(rd+ri)/2<0.97
rat={}
for m in ms:
    s=[(1-ri[k])/(1-rd[k]) for k,p in enumerate(P) if p[0]==m and mask[k] and rd[k]<.999]
    if len(s)>=3: rat[m]=np.median(s)
v=np.array(list(rat.values()));print('ratio',np.median(v),(v<1).sum(),len(v),stats.wilcoxon(v-1).pvalue)
# dedupe siblings: drop families by name stem
def stem(m):return re.sub(r'[-_.]?(v?\d[\d.]*|merged|hf|base)$','',m.split('/')[-1].lower())
fam={}
for m in ms:fam.setdefault(stem(m),[]).append(m)
print(len(fam),{k:v for k,v in fam.items() if len(v)>1})
keep=[v[0] for v in fam.values()]
k2=np.array([p[0] in keep for p in P]);print('dedupe: n',k2.sum(),'mean',df[k2].mean(),'frac',(df[k2]>0).mean())
mk=np.array([np.mean(df[[p[0]==m for p in P]]) for m in keep]);print('dedupe model mean',mk.mean(),(mk>0).sum(),len(keep))
# leave-one-out extreme
print('LOO range',min(np.delete(mn,i).mean() for i in range(len(mn))),max(np.delete(mn,i).mean() for i in range(len(mn))))
# f16 baseline
f16=np.array(['float16' in (d[p[0]][D].get('note','')) for p in P]);print('f16 rows',f16.sum(),df[f16].mean(),df[~f16].mean())
# by compression bin
c=np.array([p[2]['compression'] for p in P])
for a,b in[(0,4),(4,64),(64,128),(128,1e9)]:
    k=(c>=a)&(c<b);print(a,b,k.sum(),df[k].mean())
# width-matched via equal-n? check by baseline-level: lower-bound with per-dataset narrow
# McNemar
from scipy.stats import binomtest
print('mcn',binomtest(int((sd&~si).sum()),int((sd&~si).sum()+(~sd&si).sum())).pvalue)
# regression to ceiling: retention vs baseline. correlate within-model df with compression controlling
