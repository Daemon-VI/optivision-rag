import json,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
both=[m for m,v in d.items() if D in v and I in v]
print('both',len(both),'docvqa only',[m for m,v in d.items() if D in v and I not in v][:5])
def rows(m,ds): return {r['label']:r for r in d[m][ds]['rows']}
res=[]
for m in both:
    a,b=rows(m,D),rows(m,I)
    sh=[l for l in a if l in b and a[l]['compression']>1.01 and abs(a[l]['compression']-b[l]['compression'])/a[l]['compression']<0.02]
    sh32=[l for l in sh if a[l]['compression']>=32]
    g=lambda L,ds,r:np.mean([r[l]['retention'] for l in L])
    res.append(dict(m=m,bD=d[m][D]['baseline'],bI=d[m][I]['baseline'],n=len(sh),n32=len(sh32),
      gall=g(sh,I,b)-g(sh,D,a),g32=(g(sh32,I,b)-g(sh32,D,a)) if sh32 else np.nan,
      rD32=g(sh32,D,a) if sh32 else np.nan, rI32=g(sh32,I,b) if sh32 else np.nan))
import pandas as pd
df=pd.DataFrame(res);df['gap']=df.bI-df.bD
print(df[['n','n32']].describe().loc[['min','max']])
print('baseline I>D',(df.gap>0).sum(),len(df),df.gap.mean(),df.bD.min(),df.bI.max())
for c in['gall','g32']:
    x=df[c].dropna();print(c,x.mean(),(x>0).sum(),len(x), 'signrank p',stats.wilcoxon(x).pvalue)
    ci=[np.mean(np.random.default_rng(i).choice(x,len(x))) for i in range(2000)];print(' boot',np.percentile(ci,[2.5,97.5]))
    k=df[c].notna();print(' spearman gap',stats.spearmanr(df.gap[k],df[c][k]))
    print(' spearman bD',stats.spearmanr(df.bD[k],df[c][k]),'bI',stats.spearmanr(df.bI[k],df[c][k]))
# sibling-drop: families by org prefix
df['org']=df.m.str.split('/').str[0]
x=df.dropna(subset=['g32']).drop_duplicates('org');print('one per org',len(x),stats.spearmanr(x.gap,x.g32),x.g32.mean())
# drop top gap
y=df.dropna(subset=['g32']).sort_values('g32');print('trim extreme',stats.spearmanr(y.gap[1:-1],y.g32[1:-1]),y.g32.mean(),y.g32.median())
# pooled
bl=np.r_[df.bD,df.bI];r=np.r_[df.rD32,df.rI32];k=~np.isnan(r)
print('pooled raw',stats.spearmanr(bl[k],r[k]))
z=lambda v:(v-np.nanmean(v))/np.nanstd(v)
print('pooled std',stats.spearmanr(np.r_[z(df.bD),z(df.bI)][k],np.r_[z(df.rD32),z(df.rI32)][k]),k.sum())
print('within D',stats.spearmanr(df.bD[df.rD32.notna()],df.rD32.dropna()),'within I',stats.spearmanr(df.bI[df.rI32.notna()],df.rI32.dropna()))
print(df.sort_values('g32')[['m','bD','bI','g32']].head(4).to_string());print(df.sort_values('g32')[['m','bD','bI','g32']].tail(3).to_string())
