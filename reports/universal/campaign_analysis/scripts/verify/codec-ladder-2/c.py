import json,glob,numpy as np
base="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
U=[]
for f in glob.glob(base+"*/wide/*.json"):
    j=json.load(open(f))
    R={r["label"]:r for r in j["rows"]}
    if 'int4' in R and 'int4(mean)' in R:
        g=lambda l,m="ndcg@5":R[l].get("retention:"+m) if l in R else None
        U.append((f.replace("\\","/").split("campaign/")[1].split("/")[0],j["dataset"],g('int4'),g('int4(mean)'),g('int8(per_vector)'),g('lloyd2(7)'),g('binary'),R['int4']['bytes_per_doc'],R['int4(mean)']['bytes_per_doc'],R['int4']['retention_ci:ndcg@5'],R['int4(mean)']['retention_ci:ndcg@5']))
print(len(U), len(glob.glob(base+"*/wide/*.json")))
a=np.array([u[2] for u in U]);b=np.array([u[3] for u in U])
print((a<.97).sum(),(a<.9).sum(),(b<.97).sum(),b.min(),np.median(a))
for i,n in((2,'int4'),(3,'mean'),(4,'int8'),(5,'lloyd'),(6,'bin')):
    x=np.array([u[i] for u in U if u[i] is not None]);print(n,len(x),(x<.99).sum())
print(np.median([u[8]-u[7] for u in U])/1024,min(u[8]-u[7] for u in U)/1024,max(u[8]-u[7] for u in U)/1024)
for u in sorted(U,key=lambda u:u[2])[:11]: print(u[0],u[1][-12:],u[2],u[3],u[9],u[10],u[8]-u[7])
print(sorted(set(u[0] for u in U if u[2]<.9)))
