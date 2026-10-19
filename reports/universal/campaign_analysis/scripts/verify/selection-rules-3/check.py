import json,glob,collections,numpy as np
from scipy.stats import binomtest
rows={}
for f in glob.glob("*/selection_rules/*.json"):
    d=json.load(open(f)); key=f.replace("\\","/").split("/")[0],d['dataset']
    for r in d['results']:
        for o in r['outcomes']:
            rows[(r['reference'],key,r['target'],r['rule'],o['seed'])]=(o,r['splits'])
def run(ref,hk,excl=None,tg=None,sel=None):
    A=[];B=[]
    keys=set(k[1:3]+(k[4],) for k in rows if k[0]==ref)
    for m,t,s in sorted(keys,key=str):
        if excl and excl(m):continue
        if tg and t!=tg: continue
        a=rows.get((ref,m,t,'calibrate() default',s));b=rows.get((ref,m,t,'lower 0.999',s))
        if a is None or b is None: continue
        a=a[0];b=b[0]
        if a.get('selected') is None or b.get('selected') is None or a.get(hk) is None or b.get(hk) is None: continue
        A.append((a['selected'],a['compression'],a[hk]<t-1e-12,m));B.append((b['selected'],b['compression'],b[hk]<t-1e-12,m))
    n=len(A);diff=[i for i in range(n) if A[i][0]!=B[i][0]]
    ga=np.exp(np.mean([np.log(A[i][1]) for i in diff]));gb=np.exp(np.mean([np.log(B[i][1]) for i in diff]))
    more=np.mean([B[i][1]>A[i][1] for i in diff])
    ma=np.mean([A[i][2] for i in diff]);mb=np.mean([B[i][2] for i in diff])
    ab=sum(1 for i in range(n) if (not A[i][2]) and B[i][2]);ba=sum(1 for i in range(n) if A[i][2] and not B[i][2])
    p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1
    print(ref,hk,tg,"n",n,"diff",len(diff),round(len(diff)/n,3),"gm def/l999",round(ga,1),round(gb,1),"more",round(more,2),"miss",round(ma,3),round(mb,3),"disc",ab,ba,"p",p)
    return A,B
for ref in['labels','baseline@1']:
    for hk in['holdout','holdout_labels']:
        run(ref,hk)
A,B=run('labels','holdout')
for tg in[.95,.97,.99]:run('labels','holdout',tg=tg)
# models-level
ms=sorted(set(a[3] for a in A))
print(len(ms),"pools")
# cluster-by-model: per-model discordance
per=collections.defaultdict(lambda:[0,0,0])
for a,b in zip(A,B):
    per[a[3][0]][2]+=1
    if (not a[2]) and b[2]:per[a[3][0]][0]+=1
    if a[2] and not b[2]:per[a[3][0]][1]+=1
tot=sorted(per.items(),key=lambda x:-x[1][0])
print([(k[:25],v) for k,v in tot[:8]])
print("models with ab>0:",sum(v[0]>0 for v in per.values()),"ba>0:",sum(v[1]>0 for v in per.values()),"of",len(per))
# sign test across models
from scipy.stats import wilcoxon
d=[v[0]-v[1] for v in per.values() if v[0]!=v[1]]
print("model-level sign",sum(x>0 for x in d),sum(x<0 for x in d),binomtest(sum(x>0 for x in d),len(d),.5).pvalue)
# drop top 3 models
top=set(k for k,_ in tot[:3])
run('labels','holdout',excl=lambda m:m[0] in top)
# per-pool-clustered: pool-level counts
pool=collections.defaultdict(lambda:[0,0])
for a,b in zip(A,B):
    pool[(a[3],)] 
