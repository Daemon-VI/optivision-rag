import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl')
d=O[(O.reference=='labels')&O.sel_ok]
cell=d.groupby(['model','dataset','rule','target']).agg(n=('met','size'),met=('met','mean'),k=('met','sum'),hmin=('hold','min'),mcomp=('comp','median'),
   ret_all_min=('ret_all','min'),ret_all_med=('ret_all','median'),cal_mean=('cal','mean')).reset_index()
cell=cell.merge(F[['model','dataset','n_queries','n_docs','baseline_ndcg5','vpd','dim']],on=['model','dataset'])
cell.to_pickle(OUT+'/cell.pkl')
for rule in ['calibrate() default','bonferroni 0.975','lower 0.999','lower 0.975 + margin 0.01']:
    x=cell[(cell.rule==rule)&(cell.met<0.9)].sort_values('met')
    print('\n###',rule,'cells met<0.9:',len(x),'of',(cell.rule==rule).sum())
    print(x[['model','dataset','target','n','met','hmin','mcomp','ret_all_min','ret_all_med','n_queries','baseline_ndcg5']].round(4).to_string())
# which split-level failures: selected pipelines
x=d[(d.rule=='calibrate() default')&(d.met==0)]
print(x[['model','dataset','target','seed','sel','cal','hold','ret_all']].round(4).to_string())
