"""A8: random-32 vs PCA-32 gap, text(SciFact) vs image(ViDoRe) models at D=128 (modality and dataset confounded). Plus codec-after-PCA for wide models."""
from common import *
from scipy import stats
rnd=pp[(pp.method=='random')&(pp.k==32)][['model','ds','ret','modality']].rename(columns={'ret':'rand'})
p32=curve[(curve.k==32)&(curve.D==128)][['model','ds','rel_merge']]
j=rnd.merge(p32,on=['model','ds']);j['gap']=j.rel_merge-j['rand']
m=j.groupby(['model','modality']).agg(rand=('rand','mean'),pca=('rel_merge','mean'),gap=('gap','mean')).reset_index()
for mod_,g in m.groupby('modality'): print(mod_,len(g),'random %.3f pca %.3f gap %.3f'%(g.rand.mean(),g.pca.mean(),g.gap.mean()))
t=m[m.modality=='text'];i=m[m.modality=='image']
print('MWU gap text>image: p=%.4g ; random text<image p=%.4g ; pca text vs image p=%.3g'%(stats.mannwhitneyu(t.gap,i.gap,alternative='greater').pvalue,stats.mannwhitneyu(t.rand,i.rand,alternative='less').pvalue,stats.mannwhitneyu(t.pca,i.pca).pvalue))
r=np.random.default_rng(2);d=[np.mean(r.choice(t.gap,len(t)))-np.mean(r.choice(i.gap,len(i))) for _ in range(5000)]
print('mean gap diff text-image %.3f CI %s'%(t.gap.mean()-i.gap.mean(),np.percentile(d,[2.5,97.5]).round(3)))
print('models with random-32 retention < 0.05:',(m.rand<0.05).sum(),'of',len(m),'; all text:',m[m.rand<0.05].modality.unique())
# within image: docvqa vs infovqa random-32
ii=j[j.modality=='image'].pivot(index='model',columns='ds',values='rand')
print('image random-32 docvqa %.3f infovqa %.3f; infovqa better in %d/%d'%(ii.docvqa.mean(),ii.infovqa.mean(),(ii.infovqa>ii.docvqa).sum(),len(ii)))
# random-32 loss related to PCA-32 loss?
print('spearman model-level random-32 vs PCA-32 retention: %.2f'%stats.spearmanr(m.rand,m.pca).statistic)
