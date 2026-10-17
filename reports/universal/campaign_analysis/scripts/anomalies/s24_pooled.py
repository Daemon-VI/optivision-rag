from common import *
import pandas as pd
from scipy import stats
P=PK+'/reports/universal/audit/'
units=[]
for fn in ('corpus_size.json','colqwen_corpus_size.json'):
    d=pd.DataFrame(json.load(open(P+fn)))
    for (ds,cf),s in d.groupby(['dataset','config']):
        r=stats.linregress(np.log2(s.corpus_size),s.retention)
        units.append(dict(src=fn[:6],ds=ds[:25],config=cf,slope=r.slope))
u=pd.DataFrame(units)
u['fam']=np.where(u.config.str.contains('binary'),np.where(u.config.str.startswith('adaptive'),'adaptive>binary','binary'),np.where(u.config.str.contains('int8'),'adaptive>int8','int4'))
rng=np.random.default_rng(3)
for f,g in u.groupby('fam'):
    bs=[rng.choice(g.slope.values,len(g)).mean() for _ in range(5000)]
    print(f"{f:16s} units={len(g)} mean slope/doubling={g.slope.mean():+.4f} 95%CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] neg units={(g.slope<0).sum()}")
gb=u[u.fam.isin(['binary','adaptive>binary'])]; bs=[rng.choice(gb.slope.values,len(gb)).mean() for _ in range(5000)]
print('binary-containing pooled',gb.slope.mean().round(4),np.percentile(bs,[2.5,97.5]).round(4),len(gb))
# binary rescue by float-prefix retention bin
x=pd.read_pickle(OUT+'/binary_gain.pkl'); lo=x[x.kind=='k/d<=1/4'].copy()
lo['bin']=pd.cut(lo.r_pre,[-0.01,0.3,0.5,0.8,0.95,1.1])
print(lo.groupby('bin',observed=True).agg(n=('diff','size'),mean_diff=('diff','mean'),share_better_by_0p05=('diff',lambda s:(s>0.05).mean()),share_CI_sep=('sep','mean')).round(3).to_string())
t=lo[lo.r_pre<0.5]; print('prefix<0.5: n',len(t),'models',t.model.nunique(),'binary better',(t['diff']>0).sum(),'sign test p',stats.binomtest(int((t['diff']>0).sum()),len(t),0.5).pvalue)
# model-clustered
mm=t.groupby('model')['diff'].mean(); bs=[rng.choice(mm.values,len(mm)).mean() for _ in range(5000)]
print('per-model mean gain (prefix<0.5):',mm.mean().round(3),np.percentile(bs,[2.5,97.5]).round(3),'models',len(mm),'models>0',(mm>0).sum())
