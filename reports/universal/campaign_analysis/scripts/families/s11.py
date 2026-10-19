from common import *
from scipy.stats import spearmanr
rng=np.random.default_rng(12)
D=pd.read_csv('s9_design.csv')
D['dim']=[info(m)['dim'] for m in D.model]; D['ldim']=np.log10(D.dim)
D['grp']=[GROUP2.get(m,'single') for m in D.model]
print(D.groupby('dim')[['binary','hierarchical_merge(0.33) > binary']].agg(['mean','count']).mul([100,1,100,1]).round(2))
for ds in ['doc','info']:
    d=D[D.ds==ds]
    r,p=spearmanr(d.ldim,d.binary); print(ds,'spearman(binary loss, log dim) %.3f p %.4f n %d'%(r,p,len(d)))
# within-group (demean by group) correlation
d=D.copy()
for c in ['ldim','binary','hierarchical_merge(0.33) > binary']: d[c+'_w']=d[c]-d.groupby(['grp','ds'])[c].transform('mean')
dd=d[d.groupby('grp').model.transform('nunique')>1]
r,p=spearmanr(dd.ldim_w,dd.binary_w); print('within-group x dataset (demeaned) spearman %.3f p %.4f n %d groups with >1 model'%(r,p,len(dd)))
# within-family dim change: qwen3.5 (320/2048/4096) and qwen3vl
for g in ['qwen3.5','qwen3vl']:
    t=D[D.grp==g].groupby(['dim','ds']).binary.mean().mul(100).round(2).unstack(); print(g); print(t)
# permutation test of dose-response via cluster permutation (shuffle dim across models)
ms=D.model.unique(); dimm=D.groupby('model').ldim.first()
def stat(dimmap):
    x=D.model.map(dimmap).values
    # pooled across ds after demeaning by ds
    xs=x-pd.Series(x).groupby(D.ds.values).transform('mean').values
    ys=D.binary.values-D.groupby('ds').binary.transform('mean').values
    return np.corrcoef(xs,ys)[0,1]
obs=stat(dimm); null=[stat(dict(zip(ms,rng.permutation(dimm.loc[ms].values)))) for _ in range(5000)]
print('pooled ds-demeaned Pearson(log dim, binary loss)=%.3f ; model-level permutation p (two-sided)=%.4f'%(obs,(np.sum(np.abs(null)>=abs(obs))+1)/5001))
# vpd robustness for hm(0.1)
print('--- merge(0.1) loss vs log10 vpd')
for drop in [None,'Cognitive-Lab/ColNetraEmbed']:
    d=D if drop is None else D[D.model!=drop]
    for ds in ['doc','info']:
        x=d[d.ds==ds]; r,p=spearmanr(x.lvpd,x['hierarchical_merge(0.1)']); print('drop',drop,ds,'spearman %.3f p %.4f n %d'%(r,p,len(x)))
# expected kept vectors = 0.1*vpd
