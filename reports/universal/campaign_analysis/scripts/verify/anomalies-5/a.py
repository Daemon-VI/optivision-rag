import json,glob,os,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
rows=[];base={}
for f in sorted(glob.glob(R+'*/wide/*.json')):
    d=json.load(open(f)); m=os.path.basename(os.path.dirname(os.path.dirname(f)))
    ds=d['dataset']; base[(m,ds)]=d['baseline']['ndcg@5']
    for r in d['rows']:
        k='retention:ndcg@5'
        if k in r and r[k] is not None:
            ci=r['retention_ci:ndcg@5']
            rows.append((m,ds,r['label'],r['group'],r[k],ci[0],ci[1],r['ndcg@5']))
print(len(rows),len(base))
ret=np.array([r[4] for r in rows]); lo=np.array([r[5] for r in rows]); hi=np.array([r[6] for r in rows])
print('>1',(ret>1).sum(),(ret>1.0001).sum(),(ret>1).mean())
print('lo>1',(lo>1).sum(),'hi<1',(hi<1).sum())
for i in np.where(lo>1)[0]: print(rows[i][:7])
print('outside',((ret<lo-1e-12)|(ret>hi+1e-12)).sum())
w=hi>lo; print('rel pos',np.mean(((ret-lo)/(hi-lo))[w]),np.std(((ret-lo)/(hi-lo))[w]))
print('zero width ne1',((hi==lo)&(ret!=1)).sum(), 'zero width',(hi==lo).sum())
for ds in ['docvqa','infovqa','scifact']:
    m=[r for r in rows if ds in r[1].lower()]
    print(ds,len(m),np.mean([r[4]>1 for r in m]))
# codec rows by label
from collections import defaultdict
g=defaultdict(list)
for r in rows:
    if r[3]=='codec': g[r[2]].append((r[0],r[4]))
for l,v in g.items():
    a=np.array([x[1] for x in v]); print(l,len(a),round(a.mean()-1,4),(a>1.0001).sum(),(a<0.9999).sum(),(abs(a-1)<=1e-4).sum())
# baselines
ms=sorted({m for m,_ in base})
dv={m:b for (m,d),b in base.items() if 'docvqa' in d}; iv={m:b for (m,d),b in base.items() if 'infovqa' in d}
c=sorted(set(dv)&set(iv)); print(len(c))
x=np.array([dv[m] for m in c]);y=np.array([iv[m] for m in c])
print(stats.spearmanr(x,y))
lg=lambda p:np.log(p/(1-p))
X=lg(y);Y=lg(x);zs=[]
for i in range(len(c)):
    k=np.arange(len(c))!=i
    b=np.polyfit(X[k],Y[k],1);res=Y[k]-np.polyval(b,X[k]);s=res.std(ddof=2)
    zs.append((Y[i]-np.polyval(b,X[i]))/s)
zs=np.array(zs);j=np.argsort(-abs(zs))[:3];print([(c[i],round(zs[i],2)) for i in j])
