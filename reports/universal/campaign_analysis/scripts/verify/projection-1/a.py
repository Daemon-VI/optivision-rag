import json,numpy as np,collections
U=json.load(open('C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/verify/projection-1/units.json'))
rng=np.random.default_rng(0)
pts=[]
for ui,u in enumerate(U):
    for k,(t,r) in u['curve'].items():
        k=int(k)
        if k<u['d']: pts.append((u['model'],u['ds'],u['d'],k,r,t))
print(len(pts),collections.Counter((p[5],p[3]) for p in pts))
mods=sorted({p[0] for p in pts})
M=np.array([p[0] for p in pts]);d=np.array([p[2] for p in pts],float);k=np.array([p[3] for p in pts],float);r=np.array([p[4] for p in pts])
def pav(x,y):
    o=np.argsort(x,kind='stable');xs=x[o];ys=y[o]
    # blocks
    v=list(ys);w=[1]*len(v);xb=list(xs)
    i=0;vals=[];ws=[]
    for yy in ys:
        vals.append(yy);ws.append(1)
        while len(vals)>1 and vals[-2]>vals[-1]:
            a=vals.pop();wa=ws.pop();vals[-1]=(vals[-1]*ws[-1]+a*wa)/(ws[-1]+wa);ws[-1]+=wa
    fit=np.repeat(vals,ws)
    return xs,fit
def pred(xs,fit,xn): return np.interp(xn,xs,fit)
def cv(mask,feat):
    errs=[]
    for mo in sorted(set(M[mask])):
        tr=mask&(M!=mo);te=mask&(M==mo)
        if feat=='k': xs,f=pav(np.log(k[tr]),r[tr]);p=pred(xs,f,np.log(k[te]))
        elif feat=='kd': xs,f=pav(np.log(k[tr]/d[tr]),r[tr]);p=pred(xs,f,np.log(k[te]/d[te]))
        else:
            X=np.c_[np.ones(tr.sum()),np.log(k[tr]),np.log(d[tr])];b=np.linalg.lstsq(X,r[tr],rcond=None)[0]
            p=np.c_[np.ones(te.sum()),np.log(k[te]),np.log(d[te])]@b
        errs+=list((p-r[te])**2)
    return np.mean(errs)
sets={'all':np.ones(len(r),bool),'wide(d>128)':d>128,'ret>0.7':r>0.7,'d>=320&r>.7':(d>128)&(r>.7)}
for n,mk in sets.items(): print(n,mk.sum(),[round(cv(mk,f),5) for f in('k','kd','ols')])
# model-cluster mean eval (equal weight per model)
# drop siblings: one model per family? use top: drop ColQwen/ColPali families -> check non128 vs 128 separately
for dd in [128]:
    mk=d==dd; print('128 only',mk.sum(),[round(cv(mk,f),5) for f in('k','kd','ols')])
# regression
mk=(1-r>0.01)&(1-r<0.5)
# unit-level: average over datasets per model? use points
def reg(mk,M_):
    X=np.c_[np.ones(mk.sum()),np.log(k[mk]),np.log(d[mk])];y=np.log(1-r[mk]);return np.linalg.lstsq(X,y,rcond=None)[0]
print('n',mk.sum(),len(set(M[mk])),reg(mk,M))
def boot(mk,B=2000):
    ms=sorted(set(M[mk]));idx={m_:np.where(mk&(M==m_))[0] for m_ in ms};out=[]
    for _ in range(B):
        s=rng.choice(len(ms),len(ms));ii=np.concatenate([idx[ms[j]] for j in s])
        X=np.c_[np.ones(len(ii)),np.log(k[ii]),np.log(d[ii])];y=np.log(1-r[ii]);out.append(np.linalg.lstsq(X,y,rcond=None)[0])
    return np.percentile(out,[2.5,97.5],axis=0)
print(boot(mk))
# only pure-measured points (no proxy)
tp=np.array([p[5] for p in pts])
m2=mk&(tp=='P');print('pure only',m2.sum(),len(set(M[m2])),reg(m2,M),)
print(boot(m2,1000))
# wide only
m3=mk&(d>128);print('wide only',m3.sum(),len(set(M[m3])),reg(m3,M));print(boot(m3,1000))
# exclude loss clip: use censored-robust: regress with loss>0.001..0.9
mk4=(1-r>0.003)&(1-r<0.9);print('wider clip',mk4.sum(),reg(mk4,M));print(boot(mk4,1000))
# leave out extreme: drop d=4096, d=2560
for drop in [4096,2560,320,640]:
    m5=mk&(d!=drop);print('drop',drop,m5.sum(),reg(m5,M))
# loss at fixed k: median by d
for kk in [32,64,128]:
    print(kk,{int(dd):(round(float(np.median(1-r[(d==dd)&(k==kk)])),3),int(((d==dd)&(k==kk)).sum())) for dd in sorted(set(d)) if ((d==dd)&(k==kk)).any()})
