from common import *
import pandas as pd
from scipy import stats
m=pd.read_pickle(OUT+'/models.pkl'); W=wide_files()
bl={}
for (fo,fn),w in W.items():
    mid=fo.replace('--','/',1); k='doc' if 'docvqa' in fn else ('info' if 'infovqa' in fn else 'sci'); bl[(mid,k)]=w['baseline']
m['kind']=np.where(m.model.isin({k[0] for k in bl if k[1]=='sci'}),'text','image')
c=m[m.rank1.notna()].copy(); c['miss']=c.pairs-c.rank1
print(c.groupby('kind').miss.value_counts().sort_index().to_string())
# image models: miss vs docvqa recall@1 (baseline)
img=c[(c.kind=='image')].copy()
img['r1doc']=[bl.get((x,'doc'),{}).get('recall@1',np.nan) for x in img.model]
img['r1inf']=[bl.get((x,'info'),{}).get('recall@1',np.nan) for x in img.model]
ok=img.dropna(subset=['r1doc'])
print('n image with baseline',len(ok))
print('spearman miss vs docvqa recall@1',stats.spearmanr(ok.miss,ok.r1doc), ' vs infovqa',stats.spearmanr(ok.miss,ok.r1inf))
txt=c[c.kind=='text'].copy(); txt['r1']=[bl[(x,'sci')]['recall@1'] for x in txt.model]
print('text spearman miss vs recall@1',stats.spearmanr(txt.miss,txt.r1))
# ColVec family
for n in ['webAI-Official/webAI-ColVec1-4b','webAI-Official/webAI-ColVec1.1-4b','webAI-Official/webAI-ColVec1.1-8b']:
    print(n, c[c.model==n][['rank1','pairs','dtype','nattempt']].values.tolist())
# fisher: 1.1-4b (9/12 twice) vs all other models pooled
others=c[~c.model.str.contains('ColVec1.1-4b')]
a=int(others.rank1.sum()); b=int(others.pairs.sum()-a); print('pooled others own-first',a,'miss',b, 'rate',a/(a+b))
# binomial prob of <=3 hits... P(rank1<=9 | p=pooled rate, n=12)
p=a/(a+b); print('P(own<=9 of 12 | p=%.3f)'%p, stats.binom.cdf(9,12,p),' P(own<=10)',stats.binom.cdf(10,12,p))
# fisher 1.1-8b + 1.1-4b (10+9+9 of 36) vs others
x=[[10+9,24-19],[a,b]]; print('ColVec1.1 (8b once, 4b once) vs others fisher',stats.fisher_exact(x))
# how many models passed with exactly 10 (minimum)
print(c[(c.rank1==10)][['model','kind','dtype']].to_string())
print("distinct image-pair sets? identical vpd lists among image models:", c[c.kind=='image'].groupby('cr_vpd_mean').size().max())
