import numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250)
rng=np.random.default_rng(2)
pv=pd.read_csv('projbin.csv'); w=pd.read_csv('wide_ret.csv')
pv=pv[pv.ds!='scifact']
nat=w[w.ds!='scifact'][['model','ds','dim','binary','float16','int4(mean)','lloyd2(7)']].copy()
nat['k']=nat.dim; nat['bin_pen']=nat.binary/nat.float16; nat['src']='native'
pj=pv[['model','ds','dim','k','bin_pen','binary','float16']].copy(); pj['src']='pca'
allr=pd.concat([nat,pj],ignore_index=True)
allr['binloss']=1-allr.bin_pen
# k=128 comparison by source dim
k128=allr[allr.k==128]
k128['dimgrp']=pd.cut(k128.dim,[0,128,400,1000,5000],labels=['128 (native, no PCA)','320 (PCA)','640 (PCA)','2048-4096 (PCA)'])
def bci(x,B=5000): 
    x=np.asarray(x); b=[rng.choice(x,len(x)).mean() for _ in range(B)]; return np.percentile(b,[2.5,97.5])
print("binary penalty (1 - ret_bin/ret_fp16) at 128-bit code width, image datasets")
for ds in ['docvqa','infovqa']:
    for g,s in k128[k128.ds==ds].groupby('dimgrp',observed=True):
        ci=bci(s.binloss); print(ds,g,'n=',len(s),'models',s.model.nunique(),'mean loss %.4f'%s.binloss.mean(),'CI',ci.round(4),'median %.4f'%s.binloss.median())
# per-model-cluster test native vs pca at k=128
for ds in ['docvqa','infovqa']:
    a=k128[(k128.ds==ds)&(k128.src=='native')].binloss; b=k128[(k128.ds==ds)&(k128.src=='pca')].binloss
    print(ds,'MWU native vs pca k=128 p=',stats.mannwhitneyu(a,b).pvalue,'n',len(a),len(b),'diff mean',round(a.mean()-b.mean(),4))
# native-128 vs same-model PCA 64? And effect of k within PCA models (monotone?)
print("Within-model k effect on binary loss (PCA rows, models with >=3 k)")
res=[]
for (m,ds),s in allr[allr.src=='pca'].groupby(['model','ds']):
    if s.k.nunique()>=3:
        r=stats.spearmanr(np.log2(s.k),s.binloss)[0]; res.append((m,ds,r))
r=pd.DataFrame(res,columns=['m','ds','rho']); print("n",len(r),"mean rho",r.rho.mean().round(3),"share rho<0",(r.rho<0).mean().round(3))
print("sign test p",stats.binomtest(int((r.rho<0).sum()),len(r.dropna()),.5).pvalue)
# binary loss vs log2 k, pooled across all (k, native/pca) by dataset
for ds in ['docvqa','infovqa']:
    s=allr[allr.ds==ds]; rr=stats.spearmanr(np.log2(s.k),s.binloss); print(ds,'pooled k vs loss rho',round(rr[0],3),'p',rr[1],'n',len(s))
    for src in ['native','pca']:
        s2=s[s.src==src]; rr=stats.spearmanr(np.log2(s2.k),s2.binloss); print('   ',src,'rho',round(rr[0],3) if s2.k.nunique()>1 else None,'n',len(s2))
# pca: loss vs source dim at matched k=128 and 160?
# table: mean binloss by k and src
t=allr.groupby(['ds','src','k']).binloss.agg(['mean','count']).round(4); print(t.to_string())
