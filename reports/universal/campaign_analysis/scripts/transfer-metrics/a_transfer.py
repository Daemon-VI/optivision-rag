import numpy as np,pandas as pd
from scipy import stats
from load import load
R,B=load()
M='ndcg@5'
rng=np.random.default_rng(0)
img=R[R.ds!='scifact']
D=img[img.ds=='docvqa'].set_index(['model','label']); I=img[img.ds=='infovqa'].set_index(['model','label'])
J=D.join(I,lsuffix='_d',rsuffix='_i',how='inner')
print('paired rows',len(J),'models',J.index.get_level_values(0).nunique(),'unpaired d',len(D)-len(J),'i',len(I)-len(J))
J['safe_d']=J['lo_'+M+'_d']>=.95; J['safe_i']=J['lo_'+M+'_i']>=.95
J['dpt']=J['ret_'+M+'_i']-J['ret_'+M+'_d']
# per-model spearman
out=[]
for m,g in J.groupby(level=0):
    rho=stats.spearmanr(g['ret_'+M+'_d'],g['ret_'+M+'_i'])[0]
    # restrict to non-saturated rows: either retention <0.99
    h=g[(g['ret_'+M+'_d']<.99)|(g['ret_'+M+'_i']<.99)]
    rho2=stats.spearmanr(h['ret_'+M+'_d'],h['ret_'+M+'_i'])[0] if len(h)>8 else np.nan
    out.append(dict(model=m,n=len(g),rho=rho,rho_nonsat=rho2,pearson=np.corrcoef(g['ret_'+M+'_d'],g['ret_'+M+'_i'])[0,1],
      safe_d=g.safe_d.sum(),safe_i=g.safe_i.sum(),both=(g.safe_d&g.safe_i).sum(),
      d_only=(g.safe_d&~g.safe_i).sum(),i_only=(~g.safe_d&g.safe_i).sum(),
      mean_ret_d=g['ret_'+M+'_d'].mean(),mean_ret_i=g['ret_'+M+'_i'].mean(),med_diff=g.dpt.median(),
      base_d=g['abs_'+M+'_d'].iloc[0]/g['ret_'+M+'_d'].iloc[0] if False else np.nan))
P=pd.DataFrame(out)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(P.round(3).to_string())
print('per-model rho: median %.3f min %.3f max %.3f ; nonsat median %.3f'%(P.rho.median(),P.rho.min(),P.rho.max(),P.rho_nonsat.median()))
bs=[np.median(rng.choice(P.rho,len(P))) for _ in range(5000)]; print('median rho boot CI',np.percentile(bs,[2.5,97.5]))
print('safe set totals: d',J.safe_d.sum(),'i',J.safe_i.sum(),'both',(J.safe_d&J.safe_i).sum(),'d_only',(J.safe_d&~J.safe_i).sum(),'i_only',(~J.safe_d&J.safe_i).sum(),'of',len(J))
a=(J.safe_d&J.safe_i).sum();b=(J.safe_d&~J.safe_i).sum();c=(~J.safe_d&J.safe_i).sum();d=(~J.safe_d&~J.safe_i).sum()
po=(a+d)/len(J);pe=((a+b)*(a+c)+(c+d)*(b+d))/len(J)**2
print('kappa',(po-pe)/(1-pe),'jaccard',a/(a+b+c))
# McNemar exact
print('mcnemar exact p',stats.binomtest(b,b+c,.5).pvalue)
# P(safe on infovqa | safe on docvqa) and converse
print('P(safe_i|safe_d)',a/(a+b),'P(safe_d|safe_i)',a/(a+c))
# model-clustered bootstrap of b/(b+c)
models=list(J.index.get_level_values(0).unique()); gm={m:J.loc[m] for m in models}
bb=[]
for _ in range(2000):
    s=rng.choice(len(models),len(models)); x=pd.concat([gm[models[i]] for i in s])
    bq=(x.safe_d&~x.safe_i).sum();cq=(~x.safe_d&x.safe_i).sum(); bb.append((bq-cq)/len(x))
print('(d_only - i_only)/n cluster boot',np.mean(bb),np.percentile(bb,[2.5,97.5]))
# which dataset stricter: ret difference
print('infovqa-docvqa retention diff: mean %.4f median %.4f ; frac i>d %.3f'%(J.dpt.mean(),J.dpt.median(),(J.dpt>0).mean()))
pm=J.groupby(level=0).dpt.median(); print('model-median diffs: n i>d',(pm>0).sum(),'of',len(pm),'sign p',stats.binomtest(int((pm>0).sum()),len(pm)).pvalue, 'wilcoxon',stats.wilcoxon(pm).pvalue)
pm2=J.groupby(level=0).dpt.mean(); print('model-mean diff mean %.4f CI'%pm2.mean(),np.percentile([rng.choice(pm2,len(pm2)).mean() for _ in range(5000)],[2.5,97.5]))
# CI width
J['w_d']=J['hi_'+M+'_d']-J['lo_'+M+'_d'];J['w_i']=J['hi_'+M+'_i']-J['lo_'+M+'_i']
print('CI width docvqa %.4f infovqa %.4f'%(J.w_d.mean(),J.w_i.mean()), 'nq', B[B.ds!='scifact'].groupby('ds').nq.mean().to_dict())
# by group: diff
print(J.groupby('group_d').agg(n=('dpt','size'),mean_diff=('dpt','mean'),med=('dpt','median'),frac_i_gt=('dpt',lambda x:(x>0).mean()),sd=('safe_d','mean'),si=('safe_i','mean')).round(3))
# diff by compression bin
J['cb']=pd.cut(J.comp_d,[0,4,8,16,32,64,128,1e5])
print(J.groupby('cb').agg(n=('dpt','size'),mean_diff=('dpt','mean'),sd=('safe_d','mean'),si=('safe_i','mean'),rho=('dpt','size')).round(3))
for cb,g in J.groupby('cb'):
    print(cb, 'rho',round(stats.spearmanr(g['ret_'+M+'_d'],g['ret_'+M+'_i'])[0],3))
# pooled rho, and rho after removing model effect (within-model rank residual)
print('pooled rho',stats.spearmanr(J['ret_'+M+'_d'],J['ret_'+M+'_i'])[0])
# leave-one-model-out prediction: predict infovqa retention of model m from docvqa retention using median across other models of same label
lab=J.reset_index().pivot(index='label',columns='model',values='ret_'+M+'_d'); labi=J.reset_index().pivot(index='label',columns='model',values='ret_'+M+'_i')
res=[]
for m in lab.columns:
    ok=lab[m].notna()&labi[m].notna()
    oth_i=labi.drop(columns=m).median(axis=1)  # other models' infovqa
    oth_d=lab.drop(columns=m).median(axis=1)
    x=pd.DataFrame(dict(own_d=lab[m],own_i=labi[m],oth_i=oth_i,oth_d=oth_d))[ok]
    res.append(dict(model=m,rho_own_d=stats.spearmanr(x.own_d,x.own_i)[0],rho_oth_i=stats.spearmanr(x.oth_i,x.own_i)[0],rho_oth_d_own_i=stats.spearmanr(x.oth_d,x.own_i)[0],mae_own_d=(x.own_d-x.own_i).abs().mean(),mae_oth_i=(x.oth_i-x.own_i).abs().mean()))
Q=pd.DataFrame(res);print(Q.describe().round(3))
print('own-docvqa vs other-models-infovqa as predictor of own infovqa: paired rho diff', (Q.rho_own_d-Q.rho_oth_i).mean(), 'wilcoxon',stats.wilcoxon(Q.rho_own_d-Q.rho_oth_i).pvalue, 'mae', (Q.mae_own_d-Q.mae_oth_i).mean(), stats.wilcoxon(Q.mae_own_d-Q.mae_oth_i).pvalue)
J.reset_index().to_pickle('J.pkl'); P.to_csv('a_permodel.csv',index=False)
