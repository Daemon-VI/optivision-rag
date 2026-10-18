import sys;sys.path.insert(0,'.')
from load import load
import numpy as np, pandas as pd
from scipy.stats import spearmanr
D=load(); K='retention:ndcg@10'; C='retention_ci:ndcg@10'
rows=[]
for m,(a,b) in D.items():
    A={r['label']:r for r in a['rows']}; B={r['label']:r for r in b['rows']}
    for l in A.keys()&B.keys():
        x,y=A[l],B[l]
        rows.append(dict(m=m.split('--')[1],label=l,group=x['group'],comp=x['compression_vs_float32'],s=x[K],n=y[K],slo=x[C][0],shi=x[C][1],nlo=y[C][0],nhi=y[C][1]))
df=pd.DataFrame(rows); df['d']=df.n-df.s
df['fam']=df.label.map(lambda l: 'single' if '>' not in l and '(' in l and not l.startswith(('hier','adapt','project')) else ('merge' if l.startswith(('hier','adapt')) and '>' not in l else ('proj' if l.startswith('project') and '>' not in l else 'chain')))
df.to_pickle('df.pkl')
print(len(df),df.label.nunique())
print('overall mean s,n,d',df.s.mean(),df.n.mean(),df.d.mean())
g=df[~df.label.isin(['float32'])].groupby('label').agg(k=('m','count'),s=('s','mean'),n=('n','mean'),d=('d','mean'),dmin=('d','min'),ci_w_s=('shi',lambda x:0),)
df['wS']=df.shi-df.slo; df['wN']=df.nhi-df.nlo
g=df.groupby('label').agg(k=('m','count'),s=('s','mean'),n=('n','mean'),d=('d','mean'),dmin=('d','min'),dmax=('d','max'),wS=('wS','mean'),wN=('wN','mean'),comp=('comp','first'))
pd.set_option('display.width',250);pd.set_option('display.max_rows',100)
print(g.sort_values('comp').round(3).to_string())
# spearman within model
rs=[]
for m,x in df.groupby('m'):
    x=x[x.label!='float32']; rs.append((m,spearmanr(x.s,x.n)[0],spearmanr(x.s[x.group!='codec'] if 0 else x.s,x.n)[1]))
print(rs)
