from load import *
import warnings; warnings.filterwarnings('ignore')
from optivision.adapters import MODELS
m=measured(); m['key']=m.model+'|'+m.dataset
m['modal']=m.dataset.map(lambda d:'text' if 'scifact' in d else 'image')
keys=sorted(m.key.unique()); N=len(keys)
kmod={k:('text' if 'scifact' in k else 'image') for k in keys}
print("pairs",N,pd.Series(kmod).value_counts().to_dict())
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)
res=[]
for lab,g in m.groupby('label'):
    n=g.key.nunique()
    if n<40: continue
    r=dict(label=lab,n=n,stages=g.stages.iloc[0],comp_med=g.comp.median(),comp_min=g.comp.min())
    for T in (.95,.97,.99):
        for kind,col in (('lo','lo'),('pt','ret')):
            r[f'{kind}{T}']=(g[col]>=T).sum()
        # by modality
        for mo in ('image','text'):
            gg=g[g.modal==mo]; r[f'lo{T}_{mo}']=(gg.lo>=T).sum(); r[f'n_{mo}']=len(gg)
    res.append(r)
D=pd.DataFrame(res); D.to_csv('defaults_table.csv',index=False)
tests=0
print("\n== best label by median compression with lower-bound coverage >= C of ALL pairs (missing=fail) ==")
for T in (.95,.97,.99):
    for C in (.90,.95,1.0):
        ok=D[D[f'lo{T}']>=np.ceil(C*N-1e-9)]
        if len(ok)==0: print(T,C,'none; best coverage', D[f'lo{T}'].max()/N, D.loc[D[f'lo{T}'].idxmax(),'label']); continue
        b=ok.sort_values('comp_med',ascending=False).head(3)
        for _,x in b.iterrows(): print(T,C,x.label,f"cov={x[f'lo{T}']}/{N}",f"comp_med={x.comp_med:.1f} min={x.comp_min:.1f}",f"img {x[f'lo{T}_image']}/{x.n_image} txt {x[f'lo{T}_text']}/{x.n_text}")
print("\n== top 25 labels by coverage at lo>=.95 with comp_med>=8")
print(D[D.comp_med>=8].sort_values(['lo0.95','comp_med'],ascending=False).head(25)[['label','n','comp_med','lo0.95','lo0.97','lo0.99','pt0.95','lo0.95_image','n_image','lo0.95_text','n_text']].to_string())
