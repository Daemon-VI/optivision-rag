import json,glob,numpy as np,collections
from scipy.stats import rankdata,chi2,binomtest
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
C=['float16','int8(per_vector)','int4(mean)','int4','lloyd2(7)','binary']
units=[]
for p in sorted(glob.glob(R+'campaign/*/wide/*.json'))+sorted(glob.glob(R+'wide/**/*.json',recursive=True)):
    try:d=json.load(open(p))
    except: continue
    if 'rows' not in d: continue
    m={r['label']:r for r in d['rows'] if r['group']=='codec'}
    if not all(c in m for c in C): continue
    dt=str(d['baseline'].get('encoder',{}).get('dtype'))
    units.append((p.replace(chr(92),"/"),d['dataset'],dt,[m[c]['retention:ndcg@5'] for c in C],[(m[c]['retention_ci:ndcg@5'][1]-m[c]['retention_ci:ndcg@5'][0])/2 for c in C]))
print(len(units))
print(collections.Counter((u[1],u[2]) for u in units))
print([u[0].split('universal/')[1] for u in units if 'campaign' not in u[0]][:10])
import os
names=[os.path.basename(u[0]) for u in units]
print('dup basenames',[n for n,c in collections.Counter(names).items() if c>1])
print('colbertv2 paths',[u[0] for u in units if 'colbertv2' in u[0]])
X=np.array([u[3] for u in units]);H=np.array([u[4] for u in units])
fp16=np.array([u[2] in('torch.float16','float16') for u in units])
ds=np.array(['scifact' if 'scifact' in u[0] else ('doc' if 'docvqa' in u[0] else 'info') for u in units])
model=np.array([os.path.basename(u[0]).rsplit('_',2)[0] if 'scifact' not in u[0] else os.path.basename(u[0]).replace('_scifact.json','') for u in units])
def tol(x,eps):
    # rank with tie tolerance: greedy - items within eps of each other tied via comparing sorted adjacent chain? use pairwise: rank = 1+ #(others lower by >eps) + 0.5*ties... 
    n=len(x); r=np.zeros(n)
    for i in range(n):
        r[i]=sum(1.0 if x[i]-x[j]>eps else (0.5 if abs(x[i]-x[j])<=eps else 0) for j in range(n) if j!=i)+1
    return r
def W(X,eps):
    n,k=X.shape
    R=np.array([tol(x,eps) if eps>0 else rankdata(x) for x in X])
    S=((R.sum(0)-R.sum(0).mean())**2).sum()
    T=0
    for r in R:
        _,c=np.unique(r,return_counts=True); T+=(c**3-c).sum()
    return 12*S/(n**2*(k**3-k)-n*T),R
for e in [0,0.002]:
    w,R=W(X,e);print('W',e,w,'meanranks',R.mean(0).round(2))
rng=np.random.default_rng(0)
um=np.unique(model);idx={m:np.where(model==m)[0] for m in um}
for e in [0,0.002]:
    bs=[]
    for _ in range(1000):
        s=np.concatenate([idx[m] for m in rng.choice(um,len(um))]);bs.append(W(X[s],e)[0])
    print('boot CI',e,np.percentile(bs,[2.5,97.5]),'n models',len(um))
for name,mask in [('fp32',~fp16),('fp16',fp16),('sci',ds=='scifact'),('doc',ds=='doc'),('info',ds=='info')]:
    print(name,mask.sum(),[round(W(X[mask],e)[0],3) for e in (0,0.002)])
# strict order
for e in [0,.002,.005,.01]:
    ok=[all(x[i]>=x[i+1]-e for i in [0,1,2,4]) for x in X]  # order f16,i8,i4m,(i4 skipped?) 
    order=[0,1,2,4,5]
    ok=[all(x[order[i]]>=x[order[i+1]]-e for i in range(4)) for x in X]
    print('weak order eps',e,sum(ok))
# pairs
def pair(a,b,e=.002):
    d=X[:,a]-X[:,b];g=(d>e).sum();l=(d<-e).sum();t=len(d)-g-l
    p=binomtest(g,g+l).pvalue if g+l else 1
    return g,t,l,round(p,4),d.mean()
print('f16-i8',pair(0,1));print('i8-i4m',pair(1,2));print('lloyd-bin',pair(4,5));print('f16-i4m',pair(0,2));print('i4m-lloyd',pair(2,4));print('i4-i4m',pair(3,2))
d=X[:,4]-X[:,5]
bs=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)];print('lb mean',d.mean(),np.percentile(bs,[2.5,97.5]))
bs=[ (lambda s:d[s].mean())(np.concatenate([idx[m] for m in rng.choice(um,len(um))])) for _ in range(2000)];print('lb cluster',np.percentile(bs,[2.5,97.5]))
print('halfwidth medians',np.median(H,0).round(4))
# fp16 identity check
print('fp16 unit f16 minus 1 max abs',np.abs(X[fp16,0]-1).max(), 'fp32',np.abs(X[~fp16,0]-1).max())
# retention>1 noise: lloyd vs binary where retention
print('mean retention',X.mean(0).round(4))
# sensitivity: one unit per model (dedupe: avg across datasets) and drop each family
Xm=np.array([X[idx[m]].mean(0) for m in um]);print('per-model W',um.size,W(Xm,0)[0],W(Xm,0.002)[0])
for cut in ['','colpali','colqwen','Sauerkraut','scifact']:
    pass
print('=====EXTRA')
d=X[:,4]-X[:,5]
for name,mask in [('fp32',~fp16),('fp16',fp16),('sci',ds=='scifact'),('doc',ds=='doc'),('info',ds=='info')]:
    dd=d[mask];print('lloyd-bin',name,mask.sum(),round(dd.mean(),4),'median',round(np.median(dd),4),'lloyd>bin',(dd>.002).sum(),'<',(dd<-.002).sum())
# per-model collapse sign test
dm=np.array([d[idx[m]].mean() for m in um]);g=(dm>.002).sum();l=(dm<-.002).sum();print('per-model lloyd>bin',g,l,binomtest(g,g+l).pvalue)
# leave-one-model-out range of lloyd-bin mean and W
ws=[];ms=[]
for m in um:
    k=np.ones(len(X),bool);k[idx[m]]=False;ws.append(W(X[k],.002)[0]);ms.append(d[k].mean())
print('LOO W',min(ws),max(ws),'LOO mean d',min(ms),max(ms))
# W for top-3 only and for 4 codecs excl int4
print('W f16,i8,i4m (0.002)',W(X[:,[0,1,2]],.002)[0],'W lloyd,binary,i4m',W(X[:,[2,4,5]],.002)[0])
print('W excl int4',W(X[:,[0,1,2,4,5]],.002)[0],W(X[:,[0,1,2,4,5]],0)[0])
# int4 vs int4(mean) by dataset
e=X[:,3]-X[:,2]
for name,mask in [('sci',ds=='scifact'),('doc',ds=='doc'),('info',ds=='info')]:
    print('i4-i4m',name,round(e[mask].mean(),4),round(np.median(e[mask]),4))
# lloyd below binary units
for i in np.where(d<-.002)[0]: print(os.path.basename(units[i][0]),round(d[i],4),'hw',H[i,4].round(3),H[i,5].round(3))
# unit-level significance: diff exceeding both half-widths
sig=(np.abs(d)>np.sqrt(H[:,4]**2+H[:,5]**2));print('lloyd-bin unit diffs exceeding combined half-width (conservative, ignores pairing):',sig.sum(),'of',len(d))
