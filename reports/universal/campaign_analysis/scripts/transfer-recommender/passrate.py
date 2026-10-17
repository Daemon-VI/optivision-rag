from lomo import *
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
isimg=(mod=='image'); 
rows=[]
for j,k in enumerate(CAND):
    r=dict(pipe=key_str(k))
    for nm,msk in (('all',np.ones(nU,bool)),('img',isimg),('txt',~isimg),('doc',ds=='vidore/docvqa_test_subsampled'),('info',ds=='vidore/infovqa_test_subsampled')):
        a=~np.isnan(R[:,j])&msk
        r['n_'+nm]=a.sum()
        for t in TARGETS: r[f'p{t}_{nm}']=(R[a,j]>=t).mean() if a.any() else np.nan
        r['comp_'+nm]=np.nanmedian(C[a,j]) if a.any() else np.nan
    rows.append(r)
P=pd.DataFrame(rows).sort_values('comp_all')
P.to_csv('passrate.csv',index=False)
cols=['pipe','n_all','comp_all','p0.95_all','p0.97_all','p0.99_all','p0.95_img','p0.95_txt','p0.97_img','p0.97_txt','p0.99_img','p0.99_txt']
print(P[cols].round(3).to_string())
