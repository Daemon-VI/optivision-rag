import pandas as pd,numpy as np
pd.set_option('display.width',250)
df=pd.read_csv('table.csv')
W=df[df.codec=='int4'].set_index('tag'); R=df.pivot_table(index='tag',columns='codec',values='ret')
U=W.join(R); U['model']=W['model']
def best_t(x,y,sign,wmiss=5):
    # flag if sign*x > t ; choose t minimizing wmiss*miss+fa, ties -> midpoint
    xs=np.sort(np.unique(sign*x)); cands=(xs[:-1]+xs[1:])/2; cands=np.append(cands,[xs[0]-1])
    best=None
    for t in cands:
        f=sign*x>t; c=wmiss*np.sum(y&~f)+np.sum(~y&f)
        if best is None or c<best[0]-1e-12: best=(c,t)
    return best[1]
def lomo(stat,sign,col,thr,wm=5):
    y=(U[col]<thr).values; x=U[stat].values; miss=fa=0; missm=[];fam=[]
    for m in U.model.unique():
        te=(U.model==m).values; t=best_t(x[~te],y[~te],sign,wm)
        f=sign*x[te]>t
        a=np.sum(y[te]&~f); b=np.sum(~y[te]&f); miss+=a; fa+=b
        if a: missm.append(m)
        if b: fam.append(m)
    # full-data threshold
    tf=best_t(x,y,sign,wm)
    return dict(stat=stat,col=col,thr=thr,n_pos=int(y.sum()),LOMO_miss=int(miss),LOMO_fa=int(fa),miss_models=','.join(missm),fa_models=','.join(fam),full_t=round(sign*tf,4))
rows=[]
for stat,sign in [('cos_mean',1),('sm1',1),('sm4',1),('maxabs',1),('qcos',1),('p2r',1),('p2rc',1),('int4_sp',-1),('binm_sp',-1)]:
    rows.append(lomo(stat,sign,'int4',0.9))
for stat,sign in [('cos_mean',1),('sm1',1),('sm4',1),('maxabs',1),('bin_sp',-1),('binm_sp',-1),('int4_sp',-1)]:
    rows.append(lomo(stat,sign,'bin',0.7))
print(pd.DataFrame(rows).to_string(index=False))
# separability margins
for s in ['cos_mean','sm1','sm4','maxabs','qcos']:
    c=U[U.int4<0.9][s]; n=U[U.int4>=0.9][s]
    print(s,'collapse min %.3f max %.3f | non-collapse max %.3f min %.3f'%(c.min(),c.max(),n.max(),n.min()))
print(U[['model','sm4','int4']].sort_values('sm4').tail(14).round(3).to_string())
