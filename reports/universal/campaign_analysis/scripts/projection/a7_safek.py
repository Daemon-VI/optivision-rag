"""A7: share of units meeting retention thresholds at given absolute k; k=128 loss vs D across wide models (permutation test)."""
from common import *
from scipy import stats
pd.set_option('display.width',250,'display.max_rows',200,'display.max_colwidth',200)
def wilson(x,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=x/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
c=curve.copy();c['Dg']=pd.cut(c.D,[0,100,130,700,3000,5000],labels=['48-64','128','320-640','2048-2560','4096'])
rows=[]
for k in (64,80,128,160,256):
    for Dg,g in c[c.k==k].groupby('Dg',observed=True):
        for thr in (0.95,0.98):
            x=(g.rel>=thr).sum();lo,hi=wilson(x,len(g));rows.append(dict(k=k,Dgroup=Dg,units=len(g),models=g.model.nunique(),thr=thr,met=int(x),share=x/len(g),wilson_lo=lo,wilson_hi=hi,min_rel=g.rel.min(),mean_rel=g.rel.mean()))
print(pd.DataFrame(rows).round(3).to_string())
# k=128 vs D, wide models
w=c[(c.k==128)&(c.D>128)].groupby(['model','D']).rel.mean().reset_index();w['loss']=1-w.rel
print(w.round(4).to_string())
rho=stats.spearmanr(np.log(w.D),w.loss);print('spearman(log D, loss@128) n=%d: %.2f; p(asymptotic)=%.4f'%(len(w),rho.statistic,rho.pvalue))
r=np.random.default_rng(0);cnt=0;N=20000
for _ in range(N):
    if stats.spearmanr(w.D.values,r.permutation(w.loss.values)).statistic>=rho.statistic: cnt+=1
print('permutation p (one-sided, models treated exchangeable) = %.4f'%((cnt+1)/(N+1)))
# group means
w['g']=pd.cut(w.D,[0,700,3000,5000],labels=['320-640','2048-2560','4096'])
for g,x in w.groupby('g',observed=True): print(g,len(x),'mean loss@128 %.4f  range %.4f-%.4f'%(x.loss.mean(),x.loss.min(),x.loss.max()))
# fp16-inference confound within D>=2048
w['fp16']=w.model.map(lambda m: bs[bs.model==m].fp16.iloc[0])
print(w[['model','D','loss','fp16']].sort_values('D').to_string())
for m in w.model: 
    mi=info[m]; print(m, '|', (mi.notes or '')[:160])
