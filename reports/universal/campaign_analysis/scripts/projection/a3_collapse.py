"""A3: does retention follow absolute k or k/D? Leave-one-model-out CV of curves; k_thr vs D."""
from common import *
from scipy import stats
pd.set_option('display.width',250,'display.max_rows',200)
def pav_fit(x,y):
    o=np.argsort(x,kind='stable');xs=x[o];ys=y[o]
    out=[]
    for v,xx in zip(ys,xs):
        out.append([v,1.0,[xx]])
        while len(out)>1 and out[-2][0]>out[-1][0]:
            b2=out.pop();b1=out.pop();n=b1[1]+b2[1];out.append([(b1[0]*b1[1]+b2[0]*b2[1])/n,n,b1[2]+b2[2]])
    ux=np.array([np.mean(b[2]) for b in out]);uy=np.array([b[0] for b in out])
    return ux,uy
c=curve.copy().reset_index(drop=True)
c['lk']=np.log(c.k);c['lf']=np.log(c.frac);c['lD']=np.log(c.D)
def cv(feat,d):
    err=[]
    mods=d.model.values
    for m in np.unique(mods):
        tr=d[mods!=m];te=d[mods==m]
        if feat in('lk','lf'):
            ux,uy=pav_fit(tr[feat].values,tr.rel.values);p=np.interp(te[feat].values,ux,uy)
        else:
            X=np.c_[np.ones(len(tr)),tr.lk,tr.lD];b=np.linalg.lstsq(X,tr.rel,rcond=None)[0];p=np.c_[np.ones(len(te)),te.lk,te.lD]@b
        err+=list((te.rel.values-p)**2)
    return np.mean(err)
for name,d in [('all',c),('k<=160',c[c.k<=160]),('rel>0.7',c[c.rel>0.7]),('wide only D>128',c[c.D>128])]:
    print(name,'n=',len(d),'models',d.model.nunique(),'| LOMO CV MSE isotonic(log k): %.6f  isotonic(log k/D): %.6f  OLS(logk,logD): %.6f'%(cv('lk',d),cv('lf',d),cv('lin2',d)),flush=True)
d=c[(1-c.rel>0.01)&(1-c.rel<0.5)].copy();d['y']=np.log(1-d.rel)
def fit(dd): X=np.c_[np.ones(len(dd)),dd.lk,dd.lD];return np.linalg.lstsq(X,dd.y,rcond=None)[0]
b=fit(d);print('log(loss)= %.2f %+.2f log k %+.2f log D (n=%d, models %d)'%(b[0],b[1],b[2],len(d),d.model.nunique()))
mods=d.model.values;um=np.unique(mods);idx={m:np.where(mods==m)[0] for m in um};r=np.random.default_rng(3);bb=[]
for _ in range(2000):
    gs=r.choice(um,len(um));ii=np.concatenate([idx[g] for g in gs]);bb.append(fit(d.iloc[ii]))
bb=np.array(bb);print('cluster-boot 95%% CI: logk %s  logD %s'%(np.percentile(bb[:,1],[2.5,97.5]).round(2),np.percentile(bb[:,2],[2.5,97.5]).round(2)))
print('(if retention depended on k/D only, coef(logD) = -coef(logk); if on absolute k only, coef(logD)=0)',flush=True)
def kthr(g,thr):
    g=g.sort_values('k');k=g.k.values;rel=np.maximum.accumulate(g.rel.values)
    if rel[0]>=thr: return k[0],'<='
    if rel[-1]<thr: return k[-1],'>'
    i=np.argmax(rel>=thr);x0,x1=np.log(k[i-1]),np.log(k[i]);y0,y1=rel[i-1],rel[i]
    return float(np.exp(x0+(thr-y0)/(y1-y0)*(x1-x0))),'='
rows=[]
for (m,ds),g in c.groupby(['model','ds']):
    for thr in (0.95,0.98,0.99):
        kk,fl=kthr(g,thr);rows.append(dict(model=m,ds=ds,D=g.D.iloc[0],thr=thr,k=kk,flag=fl))
kt=pd.DataFrame(rows);kt.to_csv('kthr.csv',index=False)
for thr in (0.95,0.98,0.99):
    t=kt[kt.thr==thr];print('\nthreshold rel>=%.2f'%thr)
    print(t.groupby('D').agg(units=('k','size'),k_med=('k','median'),k_min=('k','min'),k_max=('k','max'),censored_low=('flag',lambda s:(s=='<=').sum()),censored_high=('flag',lambda s:(s=='>').sum())).round(0).to_string())
    ex=t[t.flag=='=']
    if len(ex)>5:
        sl=np.polyfit(np.log(ex.D),np.log(ex.k),1)[0];rho=stats.spearmanr(ex.D,ex.k).statistic
        mm=ex.model.values;um=np.unique(mm);ix={m:np.where(mm==m)[0] for m in um};sb=[];r=np.random.default_rng(5)
        for _ in range(2000):
            ii=np.concatenate([ix[g] for g in r.choice(um,len(um))]);dd=ex.iloc[ii]
            if dd.D.nunique()>1: sb.append(np.polyfit(np.log(dd.D),np.log(dd.k),1)[0])
        print('uncensored n=%d models=%d: k_thr ~ D^%.2f, CI %s, spearman %.2f'%(len(ex),len(um),sl,np.percentile(sb,[2.5,97.5]).round(2),rho),flush=True)
