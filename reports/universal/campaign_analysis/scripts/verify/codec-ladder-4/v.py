import json,numpy as np
from scipy import stats
rng=np.random.default_rng(0)
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
D='vidore/docvqa_test_subsampled';I='vidore/infovqa_test_subsampled'
codecs={'float16':'float16','int8':'int8(per_vector)','int4':'int4','int4(mean)':'int4(mean)','lloyd2':'lloyd2(7)','binary':'binary'}
comp={'float16':2,'int8':3.938,'int4':7.758,'int4(mean)':7.757,'lloyd2':16,'binary':32}
M={}
for m,v in d.items():
    if D in v and I in v:
        r={}
        for ds in (D,I):
            rows={x['label']:x for x in v[ds]['rows']}
            if not all(c in rows for c in codecs.values()): break
            r[ds]={k:(rows[l]['retention'],rows[l]['retention_interval'][0]) for k,l in codecs.items()}
        else: M[m]=r
print(len(M),'models; with both ds',sum(1 for m,v in d.items() if D in v and I in v))
ms=sorted(M)
def sp(a,b):return stats.spearmanr(a,b)[0]
def boot(a,b,n=2000):
    a=np.array(a);b=np.array(b);o=[]
    for _ in range(n):
        i=rng.integers(0,len(a),len(a));o.append(sp(a[i],b[i]))
    return np.nanpercentile(o,[2.5,97.5])
def corr(ms,tag):
    print(tag,len(ms))
    for c in codecs:
        a=[M[m][D][c][0] for m in ms];b=[M[m][I][c][0] for m in ms]
        print(' ',c,round(sp(a,b),3),boot(a,b).round(2))
corr(ms,'all')
# loss
a=np.array([1-M[m][D]['binary'][0] for m in ms]);b=np.array([1-M[m][I]['binary'][0] for m in ms])
print('loss',a.mean(),b.mean(),(a-b).mean(),(a>b).sum(),stats.wilcoxon(a,b).pvalue)
# family clusters: org prefix
org=lambda m:m.split('/')[0]
from collections import Counter
print(Counter(org(m) for m in ms))
# one per org
seen=set();sub=[]
for m in ms:
    if org(m) not in seen: seen.add(org(m));sub.append(m)
corr(sub,'one per org')
# drop top-influence: exclude lowest/highest binary
for ex in range(3):
    pass
a=np.array([M[m][D]['binary'][0] for m in ms]);b=np.array([M[m][I]['binary'][0] for m in ms])
print('min binary',sorted(zip(a,b,ms))[:5])
keep=[m for m in ms if min(M[m][D]['binary'][0],M[m][I]['binary'][0])>0.9]
corr(keep,'binary>0.9 both')
# leave-one-out range
loo=[sp(np.delete(a,i),np.delete(b,i)) for i in range(len(a))];print('LOO',min(loo),max(loo))
# noise ceiling: split? use retention CI width
w=[ (M[m][D]['binary'][0]-M[m][D]['binary'][1]) for m in ms];print('median half-CI binary doc',np.median(w))
# between-model sd vs CI
print('sd binary doc',a.std(),'info',b.std())
# selection
for tau in (0.99,):
    ch=[]
    for m in ms:
        pick=[]
        for ds in (D,I):
            for c in ['float16','int8','int4','int4(mean)','lloyd2','binary']:
                pass
            ok=[c for c in codecs if M[m][ds][c][0]>=tau]
            pick.append(max(ok,key=lambda c:comp[c]) if ok else None)
        ch.append(pick)
    print('agree',sum(p[0]==p[1] for p in ch),len(ch),Counter(p[0] for p in ch),Counter(p[1] for p in ch))
# baseline float16 inference models
