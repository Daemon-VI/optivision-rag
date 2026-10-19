"""A5: dataset (docvqa vs infovqa) effect on projection loss within model; baseline quality / family associations."""
from common import *
from scipy import stats
pd.set_option('display.width',250,'display.max_rows',200)
c=curve[curve.modality=='image'].copy()
w=c.pivot_table(index=['model','D','k'],columns='ds',values='rel').dropna().reset_index()
w['d_info_minus_doc']=w.infovqa-w.docvqa
ntests=0
print('Paired (same model, same k): rel(infovqa) - rel(docvqa); loss is smaller on infovqa when positive')
for k,g in w.groupby(['D','k']):
    if len(g)<2: continue
    print(int(k[0]),int(k[1]),'n=%d mean diff %.4f  infovqa>docvqa in %d/%d'%(len(g),g.d_info_minus_doc.mean(),(g.d_info_minus_doc>0).sum(),len(g)))
# pooled: normalise by using loss ratio per (model,k): loss_doc / loss_info is noisy; use diff by k-bin
w['regime']=pd.cut(w.k/1.0,[0,40,70,100,200,5000],labels=['k<=40','41-70','71-100','101-200','>200'])
for r,g in w.groupby('regime',observed=True):
    e,lo,hi=cluster_boot(g.d_info_minus_doc,g.model,B=3000)
    wp=stats.wilcoxon(g.d_info_minus_doc).pvalue if len(g)>5 else np.nan
    print('regime',r,'n=%d models=%d mean diff %.4f CI [%.4f,%.4f] frac infovqa better %.2f wilcoxon p=%.2g'%(len(g),g.model.nunique(),e,lo,hi,(g.d_info_minus_doc>0).mean(),wp))
# baseline ndcg vs rel at k=64 (128-d image) and k=80,128 for wide
b=bs.set_index(['model','ds']).baseline
c['base']=[b[(m,d)] for m,d in zip(c.model,c.ds)]
for (D,k),g in c.groupby(['D','k']):
    if len(g)>=20:
        for ds,gg in g.groupby('ds'):
            r=stats.spearmanr(gg.base,gg.rel);print('D=%d k=%d %s n=%d spearman(baseline ndcg, rel)=%.2f p=%.3f'%(D,k,ds,len(gg),r.statistic,r.pvalue))
# image vs text at k=64 (confounded)
p=curve[(curve.D==128)&(curve.k==64)]
for m,g in p.groupby('modality'):
    print(m,len(g),g.model.nunique(),'mean rel %.4f'%g.rel.mean())
img=p[p.modality=='image'];txt=p[p.modality=='text']
print('Mann-Whitney image vs text (units, confounded with dataset): p=%.3g'%stats.mannwhitneyu(img.rel,txt.rel).pvalue)
print(curve[(curve.D==128)&(curve.k==64)].groupby('ds').rel.agg(['mean','std','count']).round(4))
print(curve[(curve.D==128)&(curve.k==32)].groupby('ds').rel.agg(['mean','std','count']).round(4))
print(curve[(curve.D==128)&(curve.k==16)].groupby('ds').rel.agg(['mean','std','count']).round(4))
