import pandas as pd, numpy as np
from scipy.stats import binomtest
pd.set_option('display.width',220)
res=pd.read_csv('lomo_summary.csv')
x=res[res.rule.isin(['min','q90'])&res.strat.isin(['global','family','knn10','knn5','modality'])]
p=x.pivot_table(index=['t','rule','strat'],columns='co',values='hit_pt')
p['drop']=p[True]-p[False]; print(p.round(3))
# sign tests: docvqa 'other' vs 'same'
D=pd.read_pickle('shift.pkl')
nt=0
for t in (0.95,0.97,0.99):
    g=D[(D.t==t)&(D.ds=='docvqa_t')].pivot(index='ui',columns='mode',values='hit')
    b=((g['same'])&(~g['other'])).sum(); c=((~g['same'])&(g['other'])).sum()
    pv=binomtest(int(b),int(b+c),0.5).pvalue if b+c else np.nan; nt+=1
    print(t,'docvqa: same-hit/other-miss',b,'other-hit/same-miss',c,'exact p',round(pv,4))
    # compression and safe-ness: among 'other' mode, median comp
