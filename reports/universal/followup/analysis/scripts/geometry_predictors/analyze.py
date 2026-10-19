import pandas as pd,numpy as np
from scipy.stats import spearmanr
from itertools import product
pd.set_option('display.width',250)
df=pd.read_csv('table.csv')
W=df[df.codec=='int4'].set_index('tag')
R=df.pivot_table(index='tag',columns='codec',values='ret'); L=df.pivot_table(index='tag',columns='codec',values='lo')
U=W.join(R); U['model']=W['model']
print('n units',len(U),'models',U.model.nunique())
geo_only=['cos_mean','p2r','p2rc','sm1','sm4','maxabs','qcos','dim']
codec_stats=['int4_sp','int4_vc','int4_rr','int4m_sp','bin_sp','bin_vc','binm_sp']
def auc(score,y): # P(score of positive > negative); y=1 collapse
    s1=score[y==1];s0=score[y==0]
    if len(s1)==0 or len(s0)==0:return np.nan
    return np.mean([(a>b)+0.5*(a==b) for a in s1 for b in s0])
for codec,thr in [('int4',0.9),('binary',0.7),('int4m',0.97)]:
    col={'int4':'int4','binary':'bin','int4m':'int4m'}[codec]
    y=(U[col]<thr).astype(int); print('\n==',codec,'collapse<',thr,'n collapse',y.sum(),'models',U.model[y==1].nunique())
    out=[]
    for s in geo_only+codec_stats:
        x=U[s].values
        sp=spearmanr(x,U[col]).correlation
        a=auc(x,y.values); a=max(a,1-a) if not np.isnan(a) else a
        # sign: AUC of the better orientation
        out.append((s,'geo' if s in geo_only else 'codec',round(sp,3),round(a,3)))
    print(pd.DataFrame(out,columns=['stat','type','spearman_vs_ret','AUC_collapse']).to_string(index=False))
