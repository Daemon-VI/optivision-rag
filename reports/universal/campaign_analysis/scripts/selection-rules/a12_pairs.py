import sys; sys.path.insert(0,'.')
from load import *
O=pd.read_pickle(OUT+'/O.pkl'); rng=np.random.default_rng(7)
d=O[O.reference=='labels'].copy(); d['logc']=np.where(d.sel_ok,np.log(d.comp),0.0); d['fail']=np.where(d.sel_ok,(d.hold<d.target).astype(float),0.0)
cell=d.groupby(['model','dataset','rule','target']).agg(logc=('logc','mean'),fail=('fail','mean')).reset_index()
P=cell.pivot_table(index=['model','dataset','target'],columns='rule',values=['logc','fail'])
def pair(a,b):
    dl=(P['logc'][a]-P['logc'][b]).dropna(); g=dl.groupby(level=0).agg(['sum','count'])
    bs=[np.exp(g['sum'].values[p].sum()/g['count'].values[p].sum()) for p in (rng.integers(0,len(g),len(g)) for _ in range(2000))]
    df=(P['fail'][a]-P['fail'][b]).dropna(); g2=df.groupby(level=0).agg(['sum','count'])
    bf=[g2['sum'].values[p].sum()/g2['count'].values[p].sum() for p in (rng.integers(0,len(g2),len(g2)) for _ in range(2000))]
    print(f"{a} / {b}: compression ratio {np.exp(dl.mean()):.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]  fail diff {df.mean():+.4f} [{np.percentile(bf,2.5):+.4f},{np.percentile(bf,97.5):+.4f}]")
pair('lower 0.999','bonferroni 0.975'); pair('bonferroni 0.975','calibrate() default'); pair('lower 0.975 + margin 0.01','bonferroni 0.975'); pair('lower 0.99','lower 0.975')
# per-model pooled met (all cells) per rule: number below 0.95
s=d[d.sel_ok]
for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.975 + margin 0.01','lower 0.99']:
    m=s[s.rule==r].groupby('model').fail.mean(); print(r,'models with pooled met<0.95:',(1-m<0.95).sum(),'of',len(m),'worst model met',round((1-m).min(),3),(1-m).idxmin())
# nominal vs realised miss
for r,c in [('lower 0.975',0.975),('lower 0.99',0.99),('lower 0.999',0.999),('bonferroni 0.975',1-0.025/40),('calibrate() default',0.999)]:
    x=s[s.rule==r]; print(r,'nominal one-sided miss',round(1-c,5),'realised held-out miss',round(x.fail.mean(),4))
