from lomo import *
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
names=[key_str(k) for k in CAND]
Q=['int4m','int4','bin','lloyd2','hm0.25+int4m','hm0.25+bin','hm0.25+lloyd2','hm0.33+bin']
qj=[names.index(q) for q in Q]
# fragility = mean retention across Q  (dim128 image only, to hold dim fixed)
isimg=(mod=='image'); d128=np.array([u['dim']==128 for u in U])
dsA='vidore/docvqa_test_subsampled'; dsB='vidore/infovqa_test_subsampled'
fr={}
for m in models:
    if minfo[m]['dim']!=128 or minfo[m]['modality']!='image': continue
    fr[m]={ds[i]:np.nanmean(R[i,qj]) for i in midx[m]}
ms=[m for m in fr if len(fr[m])==2]
a=np.array([fr[m][dsA] for m in ms]); b=np.array([fr[m][dsB] for m in ms])
rho,p=spearmanr(a,b); print('n models',len(ms),'cross-dataset spearman of fragility (dim128 image)',round(rho,3),p)
# bootstrap
rg=np.random.default_rng(2); bs=[]
for _ in range(2000):
    s=rg.integers(0,len(ms),len(ms)); 
    if len(set(a[s]))>1: bs.append(spearmanr(a[s],b[s])[0])
print('CI',np.percentile(bs,[2.5,97.5]))
# per-pipeline cross-dataset correlation of retention (residual after removing pipeline mean is same as raw within pipeline)
cors=[]
for j in range(nC):
    x=[];y=[]
    for m in ms:
        ia,ib=[i for i in midx[m] if ds[i]==dsA][0],[i for i in midx[m] if ds[i]==dsB][0]
        if not np.isnan(R[ia,j]) and not np.isnan(R[ib,j]): x.append(R[ia,j]); y.append(R[ib,j])
    if len(x)>=20: cors.append((names[j],spearmanr(x,y)[0],len(x)))
cors.sort(key=lambda c:c[1])
print('pipeline-level cross-dataset spearman:',[(n,round(r,2)) for n,r,_ in cors if n in Q])
# does fragility depend on f16 inference / vpd / family
m_f16={m:minfo[m]['size'] for m in ms}
f16flag={m:any(u['f16'] for u in U if u['model']==m) for m in ms}
fv=np.array([(fr[m][dsA]+fr[m][dsB])/2 for m in ms])
fl=np.array([f16flag[m] for m in ms])
print('f16-inference models',fl.sum(),'mean fragility ret',fv[fl].mean().round(4),'vs fp32',fv[~fl].mean().round(4),'MWU p',mannwhitneyu(fv[fl],fv[~fl]).pvalue)
vpd=np.array([minfo[m]['vpd'] for m in ms]); sz=np.array([minfo[m]['size'] for m in ms])
print('spearman fragility vs vpd',spearmanr(fv,vpd)); print('vs size',spearmanr(fv,sz))
famv=pd.Series(fv,index=[minfo[m]['family'] for m in ms]).groupby(level=0).agg(['mean','std','count']); print(famv.round(4))
# between-family share of variance
tot=fv.var(ddof=0); fam=np.array([minfo[m]['family'] for m in ms])
within=sum(((fv[fam==f]-fv[fam==f].mean())**2).sum() for f in set(fam))/len(fv)
print('share of fragility variance between families',round(1-within/tot,3))
# permutation test for that
obs=1-within/tot; cnt=0
for _ in range(5000):
    pf=rg.permutation(fam); w=sum(((fv[pf==f]-fv[pf==f].mean())**2).sum() for f in set(pf))/len(fv); cnt+= (1-w/tot)>=obs
print('perm p',(cnt+1)/5001)
for m in sorted(ms,key=lambda m:(fr[m][dsA]+fr[m][dsB])/2)[:12]: print(m, round(fr[m][dsA],3), round(fr[m][dsB],3), f16flag[m], minfo[m]['family'])
