from common import *
import pandas as pd
from scipy import stats
df=pd.read_pickle(OUT+'/sel.pkl'); L=df[df.ref=='labels']
print(L.groupby('ds').model.nunique().to_dict(), 'files', L.groupby(['model','ds']).ngroups)
d=L[L.rule=='calibrate() default']
print(d[d.met<0.9][['model','ds','target','selected','met','hm','hmin','comp']].to_string())
# default rule met by ds
print(d.groupby(['ds','target']).met.mean().round(3).unstack())
# point rule: bootstrap CI over (model) of mean met
p=L[(L.rule=='point')]
rng=np.random.default_rng(1)
for t in (0.95,0.97,0.99):
    g=p[p.target==t].groupby('model').met.mean()
    bs=[rng.choice(g.values,len(g)).mean() for _ in range(4000)]
    print('point',t,'mean met',round(g.mean(),3),'CI',np.percentile(bs,[2.5,97.5]).round(3),'frac files met==1', round((p[p.target==t].met==1).mean(),3),'n models',len(g))
# lower .975: met < .975?
q=L[L.rule=='lower 0.975']
for t in (0.95,0.97,0.99):
    g=q[q.target==t]; print('lower0.975',t,'mean met',round(g.met.mean(),3),'share of files with met<0.975',round((g.met<0.975).mean(),3),'n',len(g))
# paired: default vs point compression cost
m=L[L.rule.isin(['point','calibrate() default'])].pivot_table(index=['model','ds','target'],columns='rule',values='comp')
m['ratio']=m['point']/m['calibrate() default']; print(m.groupby('target').ratio.median().round(2).to_dict())
# selected label for 'point' target .99: top labels
print(L[(L.rule=='point')&(L.target==0.99)].selected.value_counts().head(8).to_string())
