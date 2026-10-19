import numpy as np,pandas as pd
from scipy import stats
J=pd.read_pickle('J.pkl'); M='ndcg@5'; rng=np.random.default_rng(1)
rd,ri=J['ret_'+M+'_d'],J['ret_'+M+'_i']
for thr in [.90,.95,.98]:
    pd_,pi=rd>=thr,ri>=thr
    b=(pd_&~pi).sum();c=(~pd_&pi).sum()
    print('point thr',thr,'d',pd_.sum(),'i',pi.sum(),'d_only',b,'i_only',c,'both',(pd_&pi).sum(),'mcnemar p',stats.binomtest(b,b+c,.5).pvalue if b+c else None)
# lower bound thr
for thr in [.90,.95,.98]:
    sd,si=J['lo_'+M+'_d']>=thr,J['lo_'+M+'_i']>=thr
    b=(sd&~si).sum();c=(~sd&si).sum(); print('lower thr',thr,'d',sd.sum(),'i',si.sum(),'d_only',b,'i_only',c)
# width-matched: widen infovqa CI to docvqa width (scale half-widths by sqrt(nq_i/nq_d)... use ratio of observed widths per row)
ratio=(J['hi_'+M+'_d']-J['lo_'+M+'_d'])/(J['hi_'+M+'_i']-J['lo_'+M+'_i']).replace(0,np.nan)
print('median CI width ratio d/i',ratio.median())
lo_i_adj=ri-(ri-J['lo_'+M+'_i'])*ratio.fillna(1)
sd=J['lo_'+M+'_d']>=.95; si=lo_i_adj>=.95
print('width-matched: d',sd.sum(),'i',si.sum(),'d_only',(sd&~si).sum(),'i_only',(~sd&si).sum())
# retention loss ratio: (1-ri)/(1-rd) on rows with rd<0.97
h=J[(rd<.97)]
lr=(1-h['ret_'+M+'_i'])/(1-h['ret_'+M+'_d'])
print('loss ratio infovqa/docvqa (rows rd<.97, n=%d): median %.3f, IQR'%(len(h),lr.median()),lr.quantile([.25,.75]).values)
pm=h.assign(lr=lr).groupby('model').lr.median(); print('per-model median ratio: median %.3f range %.3f-%.3f; n<1: %d/%d'%(pm.median(),pm.min(),pm.max(),(pm<1).sum(),len(pm)))
print('wilcoxon pm-1',stats.wilcoxon(pm-1).pvalue)
# ratio by group
print(h.assign(lr=lr).groupby('group_d').lr.agg(['size','median']).round(3))
# leave-one-model-out wilcoxon fix
lab=J.pivot(index='label',columns='model',values='ret_'+M+'_d'); labi=J.pivot(index='label',columns='model',values='ret_'+M+'_i')
res=[]
for m in lab.columns:
    ok=lab[m].notna()&labi[m].notna()
    oi=labi.drop(columns=m).median(axis=1)[ok]; od=lab.drop(columns=m).median(axis=1)[ok]
    res.append(dict(model=m,own=stats.spearmanr(lab[m][ok],labi[m][ok])[0],oth=stats.spearmanr(oi,labi[m][ok])[0],
      mae_own=(lab[m][ok]-labi[m][ok]).abs().mean(),mae_oth=(oi-labi[m][ok]).abs().mean(),mae_othd=(od-labi[m][ok]).abs().mean()))
Q=pd.DataFrame(res).dropna()
print(len(Q),'models w/ label overlap; rho own-docvqa %.3f vs others-infovqa %.3f; wilcoxon p=%.3g'%(Q.own.mean(),Q.oth.mean(),stats.wilcoxon(Q.own-Q.oth).pvalue))
print('MAE own-docvqa %.4f vs others-infovqa %.4f p=%.3g; others-docvqa %.4f'%(Q.mae_own.mean(),Q.mae_oth.mean(),stats.wilcoxon(Q.mae_own-Q.mae_oth).pvalue,Q.mae_othd.mean()))
print('n labels per model min', lab.notna().sum().min())
print('models where own-docvqa beats others-infovqa on rho',(Q.own>Q.oth).sum(),'of',len(Q))
# Excluding saturated families
ex=Q[~Q.model.str.contains('EVIE|Vultron|webAI')]
print('non-saturated models n',len(ex),'rho own %.3f oth %.3f'%(ex.own.mean(),ex.oth.mean()))
