import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl')
rng=np.random.default_rng(1)
RULES=['point','lower 0.975','lower 0.99','lower 0.999','bonferroni 0.975','lower 0.975 + margin 0.01','calibrate() default']
d=O[O.reference=='labels'].copy()
d['logc']=np.where(d.sel_ok,np.log(d.comp),0.0); d['fail']=np.where(d.sel_ok,(d.hold<d.target).astype(float),0.0)
cell=d.groupby(['model','dataset','rule','target']).agg(logc=('logc','mean'),fail=('fail','mean'),n=('sel_ok','size')).reset_index()
P=cell.pivot_table(index=['model','dataset','target'],columns='rule',values=['logc','fail'])
def boot(x,models,B=2000):
    # x: Series indexed by (model,dataset,target); cluster bootstrap over models
    g=x.groupby(level=0).agg(['sum','count'])
    s=g['sum'].values;c=g['count'].values;idx=np.arange(len(g))
    out=[]
    for b in range(B):
        p=rng.integers(0,len(g),len(g)); out.append(s[p].sum()/c[p].sum())
    return x.mean(),np.percentile(out,[2.5,97.5])
print('== paired log-compression ratio vs calibrate() default (exp -> ratio), cluster bootstrap over 32 models; and met difference')
rows=[]
for r in RULES[:-1]:
    dl=(P['logc'][r]-P['logc']['calibrate() default']).dropna()
    m,ci=boot(dl,None); 
    df_=(P['fail'][r]-P['fail']['calibrate() default']).dropna(); fm,fci=boot(df_,None)
    # sign test across cells
    pos=(dl>1e-9).sum();neg=(dl<-1e-9).sum()
    rows.append(dict(rule=r,ratio=np.exp(m),lo=np.exp(ci[0]),hi=np.exp(ci[1]),cells=len(dl),more=pos,less=neg,dfail=fm,dfail_lo=fci[0],dfail_hi=fci[1]))
print(pd.DataFrame(rows).round(4).to_string())
# by dataset (image vs scifact) failure rates (selected only) per rule with model-cluster CI
print('== split-level failure rate (held-out<target, selected) by dataset')
s=d[d.sel_ok]
for ds in ['docvqa','infovqa','scifact']:
    for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.99','lower 0.975','point']:
        x=s[(s.dataset==ds)&(s.rule==r)]; g=x.groupby('model').fail.agg(['sum','count'])
        bs=[];
        for b in range(1000):
            p=rng.integers(0,len(g),len(g)); bs.append(g['sum'].values[p].sum()/g['count'].values[p].sum())
        print(f"{ds:8s} {r:22s} models={len(g):2d} fail={x.fail.mean():.4f} CI[{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}] gm_comp={np.exp(x.logc.mean()):.1f}")
# model-level: per-model fail rate (default) heterogeneity, permutation test
x=s[s.rule=='calibrate() default']
g=x.groupby('model').fail.agg(['sum','count']); print('default per-model fail counts nonzero:',(g['sum']>0).sum(),'of',len(g)); print(g.sort_values('sum',ascending=False).head(8))
