from lomo import *
import pandas as pd
from scipy.stats import wilcoxon
names=[key_str(k) for k in CAND]
dsA='vidore/docvqa_test_subsampled'; dsB='vidore/infovqa_test_subsampled'
img2=[m for m in models if len(midx[m])==2]
rg=np.random.default_rng(4)
print('paired docvqa - infovqa retention, per pipeline, over models (all image models with both datasets); Wilcoxon p, n tests =',nC)
rows=[]
for j,n in enumerate(names):
    d=[]
    for m in img2:
        ia=[i for i in midx[m] if ds[i]==dsA][0]; ib=[i for i in midx[m] if ds[i]==dsB][0]
        if not np.isnan(R[ia,j]) and not np.isnan(R[ib,j]): d.append(R[ia,j]-R[ib,j])
    d=np.array(d)
    if len(d)<20 or np.allclose(d,0): continue
    bs=[d[rg.integers(0,len(d),len(d))].mean() for _ in range(1000)]
    rows.append((n,len(d),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),(d<0).mean(),wilcoxon(d).pvalue))
T=pd.DataFrame(rows,columns=['pipe','n','mean_diff','lo','hi','share_doc_lower','p']).sort_values('mean_diff')
T['p_bonf']=np.minimum(T.p*len(T),1)
pd.set_option('display.width',200); pd.set_option('display.max_rows',100)
print(T.round(4).to_string(index=False))
print('pipelines with docvqa lower in Wilcoxon p_bonf<.05:',(T.p_bonf<.05).sum(),'of',len(T))
# heterogeneity: between-model variance vs mean sampling variance, dim128 image per dataset
print('heterogeneity ratio var_between / mean(se^2), dim128 image, per dataset (shared-query noise makes se^2 an upper-ish bound)')
for dsn in (dsA,dsB):
    out=[]
    for j,n in enumerate(names):
        idx=[i for i in range(nU) if ds[i]==dsn and U[i]['dim']==128 and not np.isnan(R[i,j])]
        if len(idx)<25: continue
        x=R[idx,j]
        se2=np.mean([((U[i]['rows'][CAND[j]]['retention_interval'][1]-U[i]['rows'][CAND[j]]['retention_interval'][0])/3.92)**2 for i in idx])
        out.append((n,np.var(x,ddof=1)/se2))
    o=pd.DataFrame(out,columns=['pipe','ratio']).set_index('pipe').ratio
    print(dsn[-25:], 'median ratio',o.median().round(2),'for', {k:round(o[k],1) for k in ['int4m','bin','lloyd2','hm0.25+bin','hm0.25+int4m','pca0.5+f32','f16','int8'] if k in o.index})
# shortfalls vs CI half width at the failures
D=pd.read_pickle('frontier.pkl')
g=D[(D.strat=='dimband_mod')&(D.q=='lo')&(D.c==0.9)&(~D.hit)]
hw=[]
for t,u,j in zip(g.t,g.ui,g.j):
    lo,hi=U[u]['rows'][CAND[j]]['retention_interval']; r=R[u,j]; hw.append((hi-lo)/2)
g=g.assign(hw=hw); g['within_ci']=g.short<=g.hw
print('failures dimband_mod lo>=.9 (3 targets pooled): n',len(g),'with shortfall <= CI half-width:',g.within_ci.sum())
