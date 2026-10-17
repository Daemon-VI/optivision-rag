import numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250)
rng=np.random.default_rng(5)
w=pd.read_csv('wide_ret.csv'); w['loss']=1-w.binary
# within-family dim contrasts (image)
img=w[w.ds!='scifact']
for fam in ['colqwen3','colqwen3.5']:
    s=img[img.family==fam]
    print(fam); print(s.groupby(['ds','dim']).agg(n=('loss','size'),models=('model','nunique'),mean=('loss','mean')).round(4))
# strict-order conformance
order=['float16','int8(per_vector)','int4(mean)','lloyd2(7)','binary']
for eps in [0.0,0.002,0.005,0.01]:
    ok=np.ones(len(w),bool)
    for a,b in zip(order[:-1],order[1:]): ok&=(w[a]-w[b]>=-eps).values
    print('weak-order float16>=int8>=int4(mean)>=lloyd2>=binary within eps',eps,':',ok.sum(),'/',len(w),'=%.2f'%ok.mean())
    # include int4 uncentred in ladder  
ok=np.ones(len(w),bool); o6=['float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary']
# CI half widths
d=table(); 
nl=d[d.codec.isin(['float16','int8(per_vector)','int4(mean)','lloyd2(7)','binary'])].assign(hw=lambda x:(x.hi-x.lo)/2)
print(nl.groupby('codec').hw.median().round(4))
# int8 or float16 identical to baseline? fraction of exact 1.0
print('share ret==1.0 exactly:',{c:round((w[c]==1.0).mean(),2) for c in CODECS})
# fp16-inference: binary loss vs fp32-inference (image)
for ds in ['docvqa','infovqa']:
    s=img[img.ds==ds]; a=s[s.fp16].loss; b=s[~s.fp16].loss
    print(ds,'fp16-inf n',len(a),'mean %.4f'%a.mean(),'fp32-inf n',len(b),'mean %.4f'%b.mean(),'MWU p=%.3f'%stats.mannwhitneyu(a,b).pvalue)
# dtype vs dim confound
print(pd.crosstab(img.fp16,img.dim>128))
# int4(mean) vs int4 median advantage and bytes
dd=table(); print(dd.pivot_table(index=['model','dataset'],columns='codec',values='bytes')[['int4','int4(mean)']].pipe(lambda t:(t['int4(mean)']-t['int4']).describe()))
# lowest-bytes codec within 1 pt of float16 retention, text vs image share binary
for ds in ['docvqa','infovqa','scifact']:
    s=w[w.ds==ds]; print(ds,'binary>=0.99:',(s.binary>=0.99).mean().round(2),'lloyd2>=0.99:',(s['lloyd2(7)']>=0.99).mean().round(2),'int4(mean)>=0.99',(s['int4(mean)']>=0.99).mean().round(2),'n',len(s))
