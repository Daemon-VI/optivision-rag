import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from scipy import stats
from load import load
R,B=load(); rng=np.random.default_rng(3); M='ret_ndcg@5'
pd.set_option('display.width',250)
# ratio of losses r1 vs ndcg@5 (nonsat)
for ds,g in R.groupby('ds'):
    s=g[g[['ret_ndcg@5','ret_recall@1']].min(1)<.9]
    lr=(1-s['ret_recall@1'])/(1-s['ret_ndcg@5']); pm=s.assign(lr=lr).groupby('model').lr.median()
    print(ds,'loss ratio recall@1/ndcg@5 (rows min ret<.9, n=%d): median %.3f, per-model median range %.2f-%.2f'%(len(s),lr.median(),pm.min(),pm.max()))
# common labels per modality
fams={}
def fam(m):
    n=m.lower()
    for k in ['colpali','colqwen','colsmol','evie','vultr','webai','sauerkraut','nomic','neomme','tomoro','netra','evoqwen','modernvbert','tsystems','colbert','lfm2','mxbai','lateon','jina','gte','neuml','reason']:
        if k in n: return k
    return n
B['fam']=B.model.map(fam)
for ds in ['docvqa','infovqa','scifact']:
    g=R[R.ds==ds]; labs=set.intersection(*[set(x.label) for _,x in g.groupby('model')])
    print(ds,'common labels',len(labs),'models',g.model.nunique())
    B_=B[B.ds==ds]
    sub=g[g.label.isin(labs)]
    agg=sub.groupby('model').agg(meanret=(M,'mean'),meanret_hi=(M,lambda x:x[sub.loc[x.index,'comp']>=16].mean()),
        safe=('lo_ndcg@5',lambda x:(x>=.95).mean()),minret=(M,'min'))
    # compression-matched: mean retention on rows with comp>=32
    agg['ret32']=sub[sub.comp>=32].groupby('model')[M].mean(); agg['ret128']=sub[sub.comp>=128].groupby('model')[M].mean()
    agg=agg.join(B_.set_index('model')[['base','nq','dtype','vpd','fam']])
    agg['fp16']=agg.dtype.str.contains('16')
    agg['lvpd']=np.log(agg.vpd)
    print(agg.sort_values('base').round(3).to_string())
    for y in ['meanret','ret32','ret128','safe']:
        r=stats.spearmanr(agg.base,agg[y]); 
        bs=[]
        for _ in range(3000):
            i=rng.integers(0,len(agg),len(agg)); 
            if agg.base.iloc[i].nunique()>2: bs.append(stats.spearmanr(agg.base.iloc[i],agg[y].iloc[i])[0])
        # family-level (average within family) 
        fa=agg.groupby('fam')[['base',y]].mean(); rf=stats.spearmanr(fa.base,fa[y])
        # partial on log vpd
        def res(a,b): 
            sl=np.polyfit(b,a,1); return a-np.polyval(sl,b)
        rp=stats.spearmanr(res(agg.base.rank().values,agg.lvpd.rank().values),res(agg[y].rank().values,agg.lvpd.rank().values))
        print(' %s: rho=%.3f p=%.3g CI(%.2f,%.2f) | family-avg (n=%d) rho=%.3f p=%.2g | partial on log vpd rho=%.3f p=%.2g | rho(vpd,y)=%.2f'%(y,r[0],r[1],*np.percentile(bs,[2.5,97.5]),len(fa),rf[0],rf[1],rp[0],rp[1],stats.spearmanr(agg.lvpd,agg[y])[0]))
    agg.to_csv('c_agg_%s.csv'%ds)
    if ds!='scifact':
        # excluding fp16 etc
        e=agg[~agg.fp16]; print(' fp32-only n=%d rho(base,ret32)=%.3f'%(len(e),stats.spearmanr(e.base,e.ret32)[0]))
# within-model across datasets: baseline difference vs retention difference
d=pd.read_csv('c_agg_docvqa.csv').set_index('model'); i=pd.read_csv('c_agg_infovqa.csv').set_index('model')
j=d.join(i,lsuffix='_d',rsuffix='_i',how='inner')
for y in ['meanret','ret32','ret128']:
    dy=j[y+'_i']-j[y+'_d']; db=j.base_i-j.base_d
    print('within-model: infovqa-docvqa',y,'mean %.4f; positive in %d/%d; rho(dbase,dy)=%.3f p=%.2g'%(dy.mean(),(dy>0).sum(),len(dy),*stats.spearmanr(db,dy)))
    # model-level: baseline of infovqa higher in all?
print('base_i>base_d in',(j.base_i>j.base_d).sum(),'of',len(j),'mean diff',(j.base_i-j.base_d).mean())
# pooled: dataset-adjusted. z-score within dataset
allagg=pd.concat([d.assign(ds='d'),i.assign(ds='i')])
for y in ['ret32']:
    z=allagg.groupby('ds')[[ 'base',y]].transform(lambda x:(x-x.mean())/x.std())
    print('pooled dataset-centred rho(base,%s)=%.3f p=%.3g n=%d'%(y,*stats.spearmanr(z.base,z[y]),len(z)))
    print('pooled raw rho',stats.spearmanr(allagg.base,allagg[y]))
# query-level mechanism: pipeline-level (label) retention vs baseline? row-level within model: not available.
# Row-level regression: ret ~ ds effect + base within ds   (image only) via lstsq on labels with comp>=32
img=R[R.ds!='scifact'].merge(B[['model','ds','base']],on=['model','ds'])
sub=img[(img.comp>=32)]
# compare matched-pipeline: for each label, rank-correlate retention with baseline across models within ds
out=[]
for ds,g in sub.groupby('ds'):
    for lab,x in g.groupby('label'):
        if len(x)>=25: out.append(dict(ds=ds,label=lab,rho=stats.spearmanr(x.base,x[M])[0],grp=x.group.iloc[0],comp=x.comp.median()))
O=pd.DataFrame(out); print('label-wise rho(base, retention) across models (comp>=32): n labels',len(O),O.groupby('ds').rho.agg(['mean','median','min','max']).round(3))
print((O.rho<0).groupby(O.ds).mean())
print(O.groupby('grp').rho.agg(['size','median']).round(3))
O.to_csv('c_labelwise.csv',index=False)
