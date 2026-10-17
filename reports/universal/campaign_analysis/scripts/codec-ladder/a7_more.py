import numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
rng=np.random.default_rng(4)
w=pd.read_csv('wide_ret.csv'); pv=pd.read_csv('projbin.csv')
w['loss']=1-w.binary
# (1) absolute drop vs baseline within dataset
w['absdrop']=w.base*(1-w.binary)
for ds in ['docvqa','infovqa','scifact']:
    s=w[w.ds==ds]
    print(ds,'n',len(s),'rho(ret loss,base)=%.3f p=%.3g'%stats.spearmanr(s.loss,s.base),'rho(abs ndcg drop,base)=%.3f p=%.3g'%stats.spearmanr(s.absdrop,s.base))
s=w[w.ds!='scifact']; 
# partial: base effect within dim=128 image & within dataset, residualised rank
for ds in ['docvqa','infovqa']:
    s=w[(w.ds==ds)&(w.dim==128)]; print(ds,'dim128 n',len(s),'rho(loss,base)=%.3f p=%.3g'%stats.spearmanr(s.loss,s.base),'rho(absdrop,base)=%.3f'%stats.spearmanr(s.absdrop,s.base)[0])
# (2) paired contrasts
nat=w[w.ds!='scifact'].set_index(['model','ds'])
p64=pv[pv.k==64].set_index(['model','ds'])
j=nat.join(p64[['bin_pen']],how='inner'); j=j[j.dim==128]
j['loss128']=1-j.binary/j.float16; j['loss64']=1-j.bin_pen
for ds in ['docvqa','infovqa']:
    s=j.xs(ds,level='ds'); d=s.loss64-s.loss128
    print('128->64 PCA bits, binary incremental loss',ds,'n',len(s),'mean %.4f'%d.mean(),'wilcoxon p=%.2g'%stats.wilcoxon(d).pvalue,'share worse %.2f'%(d>0).mean())
p128=pv[(pv.k==128)&(pv.dim>128)].set_index(['model','ds'])
j2=nat.join(p128[['bin_pen']],how='inner')
j2['loss_nat']=1-j2.binary/j2.float16; j2['loss128']=1-j2.bin_pen
for ds in ['docvqa','infovqa']:
    s=j2.xs(ds,level='ds'); d=s.loss128-s.loss_nat
    print('native high-d -> PCA128, binary incremental loss',ds,'n',len(s),'native %.4f pca128 %.4f diff %.4f'%(s.loss_nat.mean(),s.loss128.mean(),d.mean()),'wilcoxon p=%.2g'%(stats.wilcoxon(d).pvalue),'share lower at 128: %.2f'%((d<0).mean()))
# absolute binary retention vs baseline at PCA128 vs native for high-d
# (3) PCA rows: incremental loss vs d/k
pa=pv.copy(); pa['loss']=1-pa.bin_pen; pa['lr']=np.log2(pa.dim/pa.k); pa['lk']=np.log2(pa.k)
for ds in ['docvqa','infovqa']:
    s=pa[pa.ds==ds]
    print(ds,'PCA rows n',len(s),'rho(loss,log2 d/k)=%.3f p=%.2g'%stats.spearmanr(s.loss,s.lr),'rho(loss,log2 k)=%.3f p=%.2g'%stats.spearmanr(s.loss,s.lk),'share binary>float16(+.002) %.2f'%((s.bin_pen>1.002).mean()))
    for lo_,hi_ in [(0,3),(3,6),(6,12)]:
        t=s[(s.lr>lo_)&(s.lr<=hi_)] if lo_>0 else s[s.lr<=hi_]
        print('   log2(d/k) in (%d,%d] n=%d mean loss %.4f'%(lo_,hi_,len(t),t.loss.mean()))
# (4) tail risk table with Wilson CI
def wilson(k,n,z=1.96):
    p=k/n; c=(p+z*z/2/n)/(1+z*z/n); h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n); return c-h,c+h
print("TAIL RISK n=88 units")
for c in CODECS:
    r=[]
    for t in [0.99,0.97,0.90]:
        k=int((w[c]<t).sum()); lo_,hi_=wilson(k,len(w)); r.append(f"<{t}: {k} [{lo_*100:.0f}-{hi_*100:.0f}%]")
    print(f"{c:18s}",'; '.join(r),' min %.3f'%w[c].min())
# (5) co-failure lloyd2 vs binary
print('spearman lloyd2 vs binary loss', stats.spearmanr(w['lloyd2(7)'],w.binary))
ft=(w['lloyd2(7)']<0.97); fb=(w.binary<0.95)
print(pd.crosstab(ft,fb,rownames=['lloyd2<0.97'],colnames=['binary<0.95']))
print('Fisher',stats.fisher_exact(pd.crosstab(ft,fb).values))
# (6) family eta^2 permutation (log loss), dataset-adjusted via within-dataset z
w['ll']=np.log(w.loss.clip(lower=1e-3)); w['z']=w.groupby('ds').ll.transform(lambda s:(s-s.mean())/s.std())
def eta(df,col='z',g='family'):
    gm=df.groupby(g)[col].transform('mean'); return 1-((df[col]-gm)**2).sum()/((df[col]-df[col].mean())**2).sum()
img=w[w.ds!='scifact'].copy()
e=eta(img); 
perm=[]
fam_by_model=img.drop_duplicates('model').set_index('model').family
for _ in range(2000):
    pm=pd.Series(rng.permutation(fam_by_model.values),index=fam_by_model.index); t=img.copy(); t['family']=t.model.map(pm); perm.append(eta(t))
print('image: family eta2 (dataset-standardised log binary loss) = %.3f, permutation p=%.4f (families=%d, models=%d)'%(e,(np.sum(np.array(perm)>=e)+1)/2001,img.family.nunique(),img.model.nunique()))
# model-level eta (model identity explains across datasets)
e2=eta(img,g='model'); print('model identity eta2 (2 datasets each) = %.3f (upper bound on repeatability)'%e2)
# (7) text models table
print(w[w.ds=='scifact'][['model','dim','base']+CODECS].round(3).to_string())
