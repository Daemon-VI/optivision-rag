import json,numpy as np,collections
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
labs=["float16","int8(per_vector)","int4(mean)","int4","hierarchical_merge(0.5)","hierarchical_merge(0.25) > int8(per_vector)","hierarchical_merge(0.25) > int4(mean)","lloyd2(7)","binary","hierarchical_merge(0.25) > binary","project(128) > int4(mean)"]
pairs=[(m,ds) for m in d for ds in d[m]]
print(len(d),len(pairs))
def wil(k,n,z=1.96):
    p=k/n;den=1+z*z/n;c=(p+z*z/2/n)/den;h=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/den;return round(c-h,2),round(min(1,c+h),2)
for T in (.95,.97,.99):
  print("T",T)
  for l in labs:
    k=n=0;comp=[]
    for m,ds in pairs:
        r=[x for x in d[m][ds]["rows"] if x["label"]==l]
        if not r: continue
        n+=1;k+=r[0]["retention_interval"][0]>=T;comp.append(r[0]["compression"])
    if n: print(f" {l:45s}{k}/{n} {wil(k,n)} comp~{np.median(comp):.1f}")
# dataset split at .95
for T in(.95,):
  for sub,f in(("image",lambda ds:"vqa" in ds),("text",lambda ds:"vqa" not in ds)):
    for l in labs[:3]+labs[5:9]:
        k=n=0
        for m,ds in pairs:
            if not f(ds):continue
            r=[x for x in d[m][ds]["rows"] if x["label"]==l]
            if r:n+=1;k+=r[0]["retention_interval"][0]>=T
        print(sub,l,k,n)
# failures for int4(mean) and int8 at .95/.97
for l,T in(("int4(mean)",.95),("int8(per_vector)",.97),("int8(per_vector)",.95),("int8(per_vector)",.99)):
  print(l,T,[(m,ds[:4],round(d[m][ds]["rows"][[x["label"] for x in d[m][ds]["rows"]].index(l)]["retention_interval"][0],4)) for m,ds in pairs if l in [x["label"] for x in d[m][ds]["rows"]] and [x for x in d[m][ds]["rows"] if x["label"]==l][0]["retention_interval"][0]<T])
# model-cluster bootstrap & drop siblings
rng=np.random.default_rng(0)
models=list(d)
def cov(l,T,ms):
    k=n=0
    for m in ms:
        for ds in d[m]:
            r=[x for x in d[m][ds]["rows"] if x["label"]==l]
            if r:n+=1;k+=r[0]["retention_interval"][0]>=T
    return k,n
for l in labs[:3]+[labs[5],labs[6],labs[8]]:
    v=[]
    for _ in range(2000):
        ms=rng.choice(models,len(models));k=n=0
        for m in ms:
            a,b=cov(l,.95,[m]);k+=a;n+=b
        v.append(k/n)
    print("clusterboot",l,np.percentile(v,[2.5,97.5]).round(3))
# per-family drop: group by org prefix
fam=collections.defaultdict(list)
for m in models: fam[m.split("/")[0]].append(m)
print({k:len(v) for k,v in fam.items() if len(v)>1})
first=[v[0] for v in fam.values()]
for l in labs[:3]+[labs[5],labs[6],labs[8]]: print("1/org",l,cov(l,.95,first))
# tolerance: intervals vs retention>=T
bad=sum(1 for m,ds in pairs for x in d[m][ds]["rows"] if x["retention_interval"][0]>x["retention"]+1e-9)
print("ci excludes point",bad)
# float16 baseline models
