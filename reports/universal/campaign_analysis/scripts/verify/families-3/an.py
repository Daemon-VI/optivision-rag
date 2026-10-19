import pandas as pd,numpy as np
from scipy import stats
df=pd.read_pickle('df.pkl'); im=df[df['mod']=='image'].copy()
im['f16']=im.ev.str.contains('float16|bfloat16|fp16',case=False)
im['gen3']=im.model.str.contains('qwen3|Qwen3|EVIE|nemotron-colembed|ColVec|webAI|Vultron',case=False)
pd.set_option('display.width',250)
w=im.pivot_table(index='model',columns='ds',values='loss')
w['dim']=im.groupby('model').dim.first(); w['fam']=im.groupby('model').fam.first(); w['f16']=im.groupby('model').f16.first(); w['gen3']=im.groupby('model').gen3.first()
w['mean']=w[['docvqa_test_subsampled','infovqa_test_subsampled']].mean(axis=1)
print(w.sort_values('dim').round(2).to_string())
for c in ['docvqa_test_subsampled','infovqa_test_subsampled','mean']:
    print(c,'all',stats.spearmanr(w[c],np.log(w.dim)))
# excluding duplicates (hf/merged/ same-weights)
dup=['colpali-v1.2-hf','colpali-v1.3-hf','colpali-v1.2-merged','colqwen2-v1.0-hf','multilingual-v1.0-merged']
wd=w[~w.index.str.contains('|'.join(dup))]
print(len(wd),stats.spearmanr(wd['mean'],np.log(wd.dim)))
# dim>128 only
h=w[w.dim>128]; print('dim>128',len(h),stats.spearmanr(h['mean'],np.log(h.dim)))
print('128 by gen3',w[w.dim==128].groupby('gen3')['mean'].agg(['mean','count']))
print('128 by f16',w[w.dim==128].groupby('f16')['mean'].agg(['mean','count']))
print('gen3 by dim>128',w.assign(hi=w.dim>128).groupby(['gen3','hi'])['mean'].agg(['mean','count']))
# permutation test on rank corr, model level
rng=np.random.default_rng(0)
x=np.log(w.dim.values);y=w['mean'].values
r0=stats.spearmanr(x,y)[0]
perm=[stats.spearmanr(rng.permutation(x),y)[0] for _ in range(10000)]
print('perm p',np.mean(np.abs(perm)>=abs(r0)))
# permutation within family-ish: only among gen3
g=w[w.gen3]; x=np.log(g.dim.values);y=g['mean'].values
r0=stats.spearmanr(x,y)[0]; perm=[stats.spearmanr(rng.permutation(x),y)[0] for _ in range(10000)]
print('gen3 only',len(g),r0,np.mean(np.abs(perm)>=abs(r0)))
# OLS with cluster bootstrap on model-mean loss
def ols(d):
    X=np.column_stack([np.ones(len(d)),d.gen3.astype(float),(d.dim>128).astype(float),d.f16.astype(float)])
    return np.linalg.lstsq(X,d['mean'].values,rcond=None)[0]
print('ols',ols(w).round(2))
bs=[];ms=w.index.values
for _ in range(3000):
    s=w.loc[rng.choice(ms,len(ms))]
    try: bs.append(ols(s))
    except: pass
bs=np.array(bs);print(np.percentile(bs,[2.5,97.5],axis=0).round(2))
# leave-one-out for extreme: drop nemotron, EVIE, colvec etc.
for drop in ['nemotron','EVIE','webAI','tomoro','Vultron','Sauerkraut']:
    s=w[~w.index.str.contains(drop)]
    print('drop',drop,len(s),ols(s).round(2),stats.spearmanr(s['mean'],np.log(s.dim))[0].round(2))
# CI width on query level for binary
print(im.assign(wid=(im.hi-im.lo)*100).wid.describe())
