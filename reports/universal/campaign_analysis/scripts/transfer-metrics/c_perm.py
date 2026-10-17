import numpy as np,pandas as pd,warnings
warnings.filterwarnings('ignore')
from scipy.stats import rankdata
from load import load
R,B=load(); rng=np.random.default_rng(4); M='ret_ndcg@5'
def fam(m):
    n=m.lower()
    for k in ['colpali','colsmol','evie','vultr','webai','nomic','neomme','tomoro','netra','evoqwen','modernvbert','tsystems','colqwen','sauerkraut','mxbai','lfm2','lateon','jina','gte','neuml','reason','colbert']:
        if k in n: return {'sauerkraut':'colqwen','tomoro':'colqwen','tsystems':'colqwen','nomic':'colqwen','evoqwen':'colqwen','netra':'colqwen'}.get(k,k)
    return n
B['fam']=B.model.map(fam)
R=R.merge(B[['model','ds','base','fam']],on=['model','ds'])
def mean_rho(Y,x):
    # Y: models x labels (nan), x: models
    out=[]
    for j in range(Y.shape[1]):
        ok=~np.isnan(Y[:,j])
        if ok.sum()<10: continue
        a=rankdata(x[ok]);b=rankdata(Y[ok,j])
        if a.std()==0 or b.std()==0: continue
        out.append(np.corrcoef(a,b)[0,1])
    return np.mean(out),len(out)
def run(ds,lo,hi,NP=500,NB=300):
    g=R[(R.ds==ds)&(R.comp>=lo)&(R.comp<hi)]
    cnt=g.groupby('label').model.nunique(); g=g[g.label.isin(cnt[cnt>=10].index)]
    piv=g.pivot(index='model',columns='label',values=M)
    meta=g.drop_duplicates('model').set_index('model').reindex(piv.index)
    Y=piv.values;x=meta.base.values;f=meta.fam.values
    obs,n=mean_rho(Y,x)
    null=np.array([mean_rho(Y,rng.permutation(x))[0] for _ in range(NP)]); p=(np.sum(np.abs(null)>=abs(obs))+1)/(NP+1)
    # fixed effects: demean within family (Y per column, x)
    Yd=Y.copy();xd=x.astype(float).copy()
    for fm in np.unique(f):
        i=f==fm; xd[i]-=xd[i].mean(); Yd[i]=Y[i]-np.nanmean(Y[i],axis=0)
    obs2,_=mean_rho(Yd,xd)
    def perm_within():
        xp=xd.copy()
        for fm in np.unique(f):
            i=np.where(f==fm)[0]; xp[i]=rng.permutation(xd[i])
        return xp
    null2=np.array([mean_rho(Yd,perm_within())[0] for _ in range(NP)]); p2=(np.sum(np.abs(null2)>=abs(obs2))+1)/(NP+1)
    bs=[]
    for _ in range(NB):
        s=rng.integers(0,len(x),len(x))
        if np.unique(x[s]).size<3: continue
        bs.append(mean_rho(Y[s],x[s])[0])
    return dict(ds=ds,comp=f'{lo}-{hi}',n_models=len(x),n_labels=n,mean_rho=obs,perm_p=p,boot_lo=np.percentile(bs,2.5),boot_hi=np.percentile(bs,97.5),within_family_rho=obs2,within_family_p=p2,n_fam=len(np.unique(f)))
res=[]
for ds in ['docvqa','infovqa','scifact']:
    for lo,hi in [(1,8),(8,32),(32,128),(128,1e6),(8,1e6)]:
        res.append(run(ds,lo,hi)); print(res[-1],flush=True)
T=pd.DataFrame(res);pd.set_option('display.width',250);print(T.round(3).to_string()); T.to_csv('c_perm.csv',index=False)
print('tests',len(T)*2)
