import json,glob,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
B=['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']
U={}
for b in B:
    for p in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
        d=json.load(open(p)); U[p]=(b,d)
print(len(U))
d0=list(U.values())[0][1]; pq=d0['per_query_ndcg@10']; print(type(pq),list(pq)[:3] if isinstance(pq,dict) else len(pq))
def row(d,l): return next(r for r in d['rows'] if r['label']==l)
rng=np.random.default_rng(0)
res=[]
for p,(b,d) in U.items():
    name=p.replace(chr(92),'/').split('/')[-1]
    res.append((name,d))
tot={}
for pair in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
  for m in ['ndcg@5','ndcg@10']:
    v=np.array([row(d,pair[0])['retention:'+m]-row(d,pair[1])['retention:'+m] for n,d in res])
    print(pair,m,len(v),(v<0).sum(),round(np.median(v),4),round(v.mean(),4),round(v.min(),4),'cells<=-.005',(v<=-.005).sum())
# healthy
for pair in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
    sel=[(row(d,pair[0])['retention:ndcg@5']-row(d,pair[1])['retention:ndcg@5']) for n,d in res if row(d,pair[1])['retention:ndcg@5']>=0.95]
    sel=np.array(sel);print('healthy',pair,len(sel),(sel<0).sum(),sel.mean().round(4),sel.std(ddof=1).round(4))
# paired
def boot(x):
    n=len(x);idx=rng.integers(0,n,(10000,n));m=x[idx].mean(1);return np.percentile(m,[2.5,97.5])
from math import comb
def sign(u,dn):
    n=u+dn
    if n==0:return 1.0
    k=min(u,dn);return min(1,2*sum(comb(n,i) for i in range(k+1))/2**n)
cnt={}
for n,d in res:
    pq=d['per_query_ndcg@10']
    for pair in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
        x=np.array(pq[pair[0]])-np.array(pq[pair[1]])
        lo,hi=boot(x); u=(x>1e-12).sum();dn=(x<-1e-12).sum();p=sign(u,dn)
        c=cnt.setdefault(pair,[0,0,0,0]);
        if hi<0:c[0]+=1
        if lo>0:c[1]+=1
        if x.mean()<0 and p<0.05:c[2]+=1
        if x.mean()<0:c[3]+=1
        if 'colpali-v1.3_docvqa' in n or ('colpali-v1.3' in n and 'docvqa' in n):print(n,pair,len(x),round(x.mean(),4),lo.round(4),hi.round(4),u,dn,p)
print(cnt)
# EVIE
for n,d in res:
    if 'EVIE' in n and 'tabfquad' in n:
        pq=d['per_query_ndcg@10']
        for l in ['int4(mean)','int4norm(mean)','int8(per_vector)']:
            r=row(d,l);print(l,r['retention:ndcg@5'],r['retention_ci:ndcg@5'])
        for a,b2 in [('int4(mean)','int4'),('int4norm(mean)','int4norm'),('int4norm(mean)','float32'),('int4norm(mean)','int4(mean)'),('int4norm(mean)','int8(per_vector)')]:
            x=np.array(pq[a])-np.array(pq[b2]);lo,hi=boot(x);print(a,b2,x.mean().round(4),lo.round(3),hi.round(3),(x>1e-12).sum(),(x<-1e-12).sum(),sign((x>1e-12).sum(),(x<-1e-12).sum()))
        print('int8 changed',(np.abs(np.array(pq['int8(per_vector)'])-np.array(pq['float32']))>1e-12).sum(), len(pq['float32']))
        print({k:v for k,v in d['norms'].items() if k in('int4',)})
