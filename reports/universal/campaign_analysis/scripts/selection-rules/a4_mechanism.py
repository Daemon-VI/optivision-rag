import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl')
d=O[(O.reference=='labels')&O.sel_ok].copy()
d['fail']=(d.hold<d.target).astype(int); d['slack']=d.ret_all-d.target; d['short']=np.maximum(d.target-d.hold,0)
d['opt']=d.cal-d.hold
print('== optimism / shortfall by rule')
g=d.groupby('rule').agg(n=('fail','size'),fail=('fail','mean'),cal=('cal','mean'),hold=('hold','mean'),opt=('opt','mean'),slack=('slack','mean'),slack_neg=('slack',lambda s:(s<0).mean()),
  short_mean_fail=('short',lambda s:s[s>0].mean()),short_p95_fail=('short',lambda s:np.percentile(s[s>0],95) if (s>0).any() else np.nan),max_short=('short','max'))
print(g.round(4).to_string())
# slack curve
bins=[-1,-0.03,-0.015,-0.005,0,0.005,0.01,0.02,0.05,1]
d['sb']=pd.cut(d.slack,bins)
print('== P(held-out < target) by slack of the chosen pipeline (all-query retention minus target), all rules pooled, labels ref')
print(d.groupby('sb',observed=True).agg(n=('fail','size'),fail=('fail','mean')).round(3).to_string())
print(pd.crosstab(d.rule,d.sb,normalize='index').round(3).to_string())
# slack-conditional failure by dataset
print(d.groupby(['dataset','sb'],observed=True).fail.agg(['size','mean']).round(3).unstack(0).to_string())
# noise sigma: dedupe by (model,dataset,seed,sel)
u=d.drop_duplicates(['model','dataset','seed','sel'])
u['res']=u.hold-u.ret_all
nz=u.groupby(['model','dataset','sel']).res.agg(['size','std']).reset_index()
nz=nz[nz['size']>=4].merge(F[['model','dataset','n_queries','baseline_ndcg5']],on=['model','dataset'])
print('candidate-level hold residual sd, n groups',len(nz))
print(nz.groupby('dataset')['std'].describe().round(4))
for ds in ['docvqa','infovqa']:
    z=nz[nz.dataset==ds]; print(ds,'spearman sd vs baseline',stats.spearmanr(z['std'],z.baseline_ndcg5))
u2=u.merge(F[['model','dataset','baseline_ndcg5']],on=['model','dataset'])
u2.to_pickle(OUT+'/u2.pkl')
