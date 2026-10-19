"""A4: PCA vs random projection at equal k."""
from common import *
from scipy import stats
pd.set_option('display.width',250,'display.max_rows',200)
rnd=pp[pp.method=='random'][['model','ds','k','D','ret','lo','hi','family','modality']].rename(columns={'ret':'rand','lo':'rlo','hi':'rhi'})
cur=curve[['model','ds','k','rel_pure','rel_merge','rel']]
j=rnd.merge(cur,on=['model','ds','k'],how='left')
j['frac']=j.k/j.D
print('random rows',len(rnd),' with pca at same k (pure):',j.rel_pure.notna().sum(),' with merge-controlled pca at same k:',j.rel_merge.notna().sum())
# (a) exact equal k, pure pca vs random (all at k/D=0.25)
a=j.dropna(subset=['rel_pure']).copy();a['diff']=a.rel_pure-a['rand']
print(a[['model','ds','D','k','rel_pure','rand','diff']].round(4).to_string())
print('PURE equal-k pairs: n=%d models=%d'%(len(a),a.model.nunique()))
e,lo,hi=cluster_boot(a['diff'],a.model,B=5000);print('mean PCA-random %.4f CI [%.4f,%.4f]'%(e,lo,hi))
print('units where PCA>random: %d/%d ; models-level (mean over ds) %d/%d'%((a['diff']>0).sum(),len(a),(a.groupby('model')['diff'].mean()>0).sum(),a.model.nunique()))
mm=a.groupby('model')['diff'].mean();print('model-level exact sign test p=%.4f'%stats.binomtest(int((mm>0).sum()),len(mm),0.5).pvalue, ' wilcoxon p=%.4f'%stats.wilcoxon(mm).pvalue)
# by D class
a['Dc']=pd.cut(a.D,[0,400,700,5000],labels=['320','640','2048-4096'])
print(a.groupby('Dc')['diff'].agg(['mean','std','count']).round(4))
# (b) 128-d: random-32 vs PCA-32 (merge-controlled)
b=j[(j.D==128)].dropna(subset=['rel_merge']).copy();b['diff']=b.rel_merge-b['rand']
print('\n128-d: PCA-32 (merge-controlled) minus random-32: n=%d models=%d'%(len(b),b.model.nunique()))
e,lo,hi=cluster_boot(b['diff'],b.model,B=5000);print('mean %.4f CI [%.4f,%.4f]; PCA wins %d/%d; min diff %.3f'%(e,lo,hi,(b['diff']>0).sum(),len(b),b['diff'].min()))
print('mean random-32 %.3f, mean PCA-32 %.3f'%(b['rand'].mean(),b.rel_merge.mean()))
# random-32 by modality/family
print(b.groupby('modality').agg(rand=('rand','mean'),pca32=('rel_merge','mean'),n=('rand','size')).round(3))
t=b.groupby(['model']).agg(rand=('rand','mean'),pca32=('rel_merge','mean'),family=('family','first'),mod=('modality','first')).sort_values('rand')
print(t.round(3).to_string())
# random retention vs absolute k, at k/D=0.25
print('\nRandom projection at k/D=0.25 by k:')
g=j.groupby('k').agg(n=('rand','size'),models=('model','nunique'),mean=('rand','mean'),std=('rand','std'),min=('rand','min'),max=('rand','max')).round(3);print(g)
# JL-style: loss vs k among D>=128
jj=j[(j.k>=32)].copy();jj['loss']=1-jj['rand']
mmod=jj.groupby(['model','k']).loss.mean().reset_index()
print('model-level mean random loss by k:',mmod.groupby('k').loss.mean().round(3).to_dict())
# random at larger D vs same k: nothing. Variation between ds
