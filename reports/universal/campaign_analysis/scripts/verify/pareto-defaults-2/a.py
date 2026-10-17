import json,numpy as np
from scipy import stats
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled';S='BeIR/scifact'
def get(m,ds,l):
    x=d[m].get(ds)
    if not x: return None
    for r in x["rows"]:
        if r["label"]==l: return r
labs=["hierarchical_merge(0.25) > int4(mean)","binary","hierarchical_merge(0.25) > project(128) > binary","hierarchical_merge(0.25) > int8(per_vector)"]
for l in labs:
  print(l)
  for ds in (D,I,S):
    rs=[(m,get(m,ds,l)) for m in d if get(m,ds,l)]
    n=len(rs); c=sum(r["retention_interval"][0]>=0.95 for m,r in rs)
    print(' ',ds[-12:],c,n)
# paired
img=[m for m in d if D in d[m] and I in d[m]]
print(len(img))
def paired(l1,l2=None):
    a=[];b=[];ms=[]
    for m in img:
        rd=get(m,D,l1);ri=get(m,I,l1)
        if rd and ri and (l2 is None or (get(m,D,l2) and get(m,I,l2))):
            if l2:
                a.append(rd["retention"]-get(m,D,l2)["retention"]);b.append(ri["retention"]-get(m,I,l2)["retention"])
            else:
                a.append(rd["retention"]);b.append(ri["retention"])
            ms.append(m)
    return np.array(a),np.array(b),ms
def rep(a,b,name):
    dd=a-b;rng=np.random.default_rng(0)
    bs=[rng.choice(dd,len(dd)).mean() for _ in range(10000)]
    print(name,len(dd),dd.mean().round(4),np.percentile(bs,[2.5,97.5]).round(4),(dd<0).sum(),stats.wilcoxon(a,b).pvalue,'median',np.median(dd))
    return dd
a,b,ms=paired("binary");dd=rep(a,b,"binary D-I")
# remove siblings/extreme
print(sorted(zip(dd.round(3),ms))[:5])
for l in ["float16","int8(per_vector)","int4","int4(mean)","lloyd2(7)"]:
    a2,b2,_=paired(l);rep(a2,b2,l)
# hm .25 > binary minus binary
hb="hierarchical_merge(0.25) > binary"
a1=[];b1=[]
for m in img:
    x=[get(m,ds,hb) for ds in (D,I)]+[get(m,ds,"binary") for ds in (D,I)]
    if all(x): a1.append(x[2]["retention"]-x[0]["retention"]);b1.append(x[3]["retention"]-x[1]["retention"])
a1=np.array(a1);b1=np.array(b1);print('hm>binary vs binary: (binary - hm>binary)?');rep(a1,b1,"bin-hmbin")
# models in doc (40) vs img list
print(len([m for m in d if D in d[m]]),len([m for m in d if I in d[m]]),len([m for m in d if S in d[m]]))
# float16 baseline notes
f16=[m for m in img if 'float16' in d[m][D]["note"]]
print(len(f16))
a,b,ms=paired("binary")
mask=np.array([m not in f16 for m in ms]);rep(a[mask],b[mask],"binary no-f16")
# family-level: drop siblings by org
org=lambda m:m.split('/')[0]
keep=[];seen=set()
for i,m in enumerate(ms):
    if org(m) not in seen: seen.add(org(m));keep.append(i)
rep(a[keep],b[keep],"binary one-per-org %d"%len(keep))
# controlling for baseline
bd=np.array([d[m][D]["baseline"] for m in ms]);bi=np.array([d[m][I]["baseline"] for m in ms])
print(np.median(bd),np.median(bi))
# coverage binary among img models that have baseline high on docvqa
for m in ms:
    pass
# matching: DocVQA models with baseline>=0.85
import itertools
hi=[(m,d[m][D]["baseline"],get(m,D,"binary")["retention"],get(m,D,"binary")["retention_interval"][0]) for m in ms if d[m][D]["baseline"]>=0.8]
print(hi)
print(sorted((round(d[m][D]["baseline"],2),round(get(m,D,"binary")["retention"],3),round(get(m,I,"binary")["retention"],3)) for m in ms))
