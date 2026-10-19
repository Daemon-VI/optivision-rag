from common import *
from scipy.stats import spearmanr, pearsonr
rng=np.random.default_rng(2)
df=long_df()
P=mat(df)
labs=[l for l in P.columns if (P.xs('doc',level='ds')[l].notna().sum()>=30) and l not in('float32','float16') and not l.endswith('> float16')]
D=1-P.xs('doc',level='ds'); I=1-P.xs('info',level='ds')
ms=[m for m in D.index if m in I.index]
rows=[]
for l in labs:
    a=D.loc[ms,l]; b=I.loc[ms,l]; ok=a.notna()&b.notna()
    a=a[ok].values;b=b[ok].values
    if len(a)<20: continue
    r,p=spearmanr(a,b)
    # bootstrap CI over models
    bs=[]
    for _ in range(1000):
        i=rng.integers(0,len(a),len(a)); 
        bs.append(spearmanr(a[i],b[i])[0])
    rows.append((l,len(a),r,p,*np.nanpercentile(bs,[2.5,97.5]),a.mean(),b.mean()))
R=pd.DataFrame(rows,columns=['label','n','rho','p','lo','hi','loss_doc','loss_info']).sort_values('rho',ascending=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(R.round(3).to_string(index=False))
print('n tests',len(R))
R.to_csv('s3_persistence.csv',index=False)
# summary: median rho
print('median rho',R.rho.median(),'share sig p<.05',(R.p<.05).mean(), 'Bonferroni',(R.p<.05/len(R)).sum())
