from lomo import *
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
df=pd.read_pickle('calib.pkl')
dimm={m:minfo[m]['dim'] for m in models}; modm={m:minfo[m]['modality'] for m in models}
df['dim']=df.model.map(dimm); df['mod']=df.model.map(modm)
df['se']=(df.p-df.y)**2
rg=np.random.default_rng(5)
def pairdiff(a,b,sub,t):
    ga=sub[(sub.strat==a)&(sub.t==t)].groupby(['model','ui','j']).se.first()
    gb=sub[(sub.strat==b)&(sub.t==t)].groupby(['model','ui','j']).se.first()
    d=(ga-gb).dropna().groupby('model').mean()  # per model mean diff
    v=d.values; bs=[v[rg.integers(0,len(v),len(v))].mean() for _ in range(2000)]
    return v.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),len(v)
print('PAIRED Brier diff (A-B) negative = A better, bootstrap over held-out models; common (model,unit,pipeline) rows')
nt=0
for t in TARGETS:
    for a,b in [('dimband_mod','global'),('dimband_mod','modality'),('samedim_mod','global'),('family','global'),('family','dimband_mod'),('knn5','dimband_mod'),('dknn5','dimband_mod'),('dimband','dimband_mod'),('knn10','dimband_mod'),('family','modality')]:
        d,lo,hi,n=pairdiff(a,b,df,t); nt+=1
        print(f't={t} {a:12s} - {b:12s}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] models={n}')
print('n paired comparisons',nt)
print('--- subset: held-out image dim128 (hardest bucket)')
sub=df[(df['dim']==128)&(df['mod']=='image')]
for t in TARGETS:
    for s in ['global','modality','dimband_mod','family','knn5','knn10','dknn5']:
        g=sub[(sub.strat==s)&(sub.t==t)]
        print(t,s,len(g),round(g.se.mean(),4),'base',round(g.y.mean(),3))
    for a,b in [('family','dimband_mod'),('knn5','dimband_mod'),('knn10','dimband_mod')]:
        d,lo,hi,n=pairdiff(a,b,sub,t); nt+=1
        print(f'   t={t} {a} - {b}: {d:+.4f} [{lo:+.4f},{hi:+.4f}] n_models={n}')
print('--- subset: held-out text dim128')
sub=df[(df['dim']==128)&(df['mod']=='text')]
for t in TARGETS:
    for s in ['global','modality','dimband_mod','family','knn5']:
        g=sub[(sub.strat==s)&(sub.t==t)]
        print(t,s,len(g),round(g.se.mean(),4),'base',round(g.y.mean(),3))
print('total tests so far',nt)
# reliability for dimband_mod
for t in TARGETS:
    g=df[(df.strat=='dimband_mod')&(df.t==t)]
    b=pd.cut(g.p,[0,.1,.3,.5,.7,.9,.97,1.0],include_lowest=True)
    print(t); print(g.groupby(b,observed=True).agg(n=('y','size'),pred=('p','mean'),obs=('y','mean')).round(3))
