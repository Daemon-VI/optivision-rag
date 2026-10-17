import json,numpy as np
from scipy import stats
exec(open('a.py').read().split("labs=")[0])
img=[m for m in d if D in d[m] and I in d[m] and get(m,D,"binary") and get(m,I,"binary")]
def boot(a,b,name):
    dd=a-b;rng=np.random.default_rng(1)
    bs=[rng.choice(dd,len(dd)).mean() for _ in range(10000)]
    print(name,len(dd),dd.mean().round(4),np.percentile(bs,[2.5,97.5]).round(4),(dd<0).sum(),'p=%.2g'%stats.wilcoxon(a,b).pvalue)
R=lambda ms,ds,l:np.array([get(m,ds,l)["retention"] for m in ms])
bd=np.array([d[m][D]["baseline"] for m in img]);bi=np.array([d[m][I]["baseline"] for m in img])
print('baseline med',np.median(bd),np.median(bi))
print(sorted(set(str(d[m][D].get("note"))[:30] for m in img)))
f16=[m for m in img if 'float16' in str(d[m][D].get('note',''))+str(d[m][I].get('note',''))]
print('f16',len(f16))
for l in ["binary","hierarchical_merge(0.25) > binary","hierarchical_merge(0.25) > project(128) > binary"]:
    boot(R(img,D,l),R(img,I,l),l)
    ms=[m for m in img if m not in f16]; boot(R(ms,D,l),R(ms,I,l),' no-f16')
    seen=set();ms=[]
    for m in img:
        o=m.split('/')[0]
        if o not in seen: seen.add(o);ms.append(m)
    boot(R(ms,D,l),R(ms,I,l),' one/org')
    dd=R(img,D,l)-R(img,I,l); i=np.argsort(dd)[:3]; ms=[m for k,m in enumerate(img) if k not in i]
    boot(R(ms,D,l),R(ms,I,l),' drop 3 most extreme')
    # baseline-matched: models where docvqa baseline >=0.85
    ms=[m for m in img if d[m][D]["baseline"]>=0.8]; 
    if len(ms)>5: boot(R(ms,D,l),R(ms,I,l),' docvqa baseline>=0.8 n=%d'%len(ms))
    # regression of diff on baseline gap and baseline D
    y=R(img,D,l)-R(img,I,l); X=np.c_[np.ones(len(y)),bd-bi]
    beta,res,_,_=np.linalg.lstsq(X,y,rcond=None); print(' intercept at zero baseline gap',beta.round(4))
# per-dataset within retention vs baseline: does D binary retention at same baseline as I match?
x=np.r_[bd,bi];y=np.r_[R(img,D,"binary"),R(img,I,"binary")];ds=np.r_[np.zeros(len(bd)),np.ones(len(bi))]
X=np.c_[np.ones(len(x)),x,ds];print('pooled binary ret ~ baseline + isInfo',np.linalg.lstsq(X,y,rcond=None)[0].round(4))
# scifact 
# Oracle compression
def oracle(ds,models):
    out=[]
    for m in models:
        c=[r["compression"] for r in d[m][ds]["rows"] if r["retention_interval"][0]>=0.95]
        out.append(max(c) if c else 1.0)
    return np.median(out),len(out)
print('oracle D',oracle(D,img),'I',oracle(I,img),'S',oracle(S,[m for m in d if S in d[m]]))
# robust-default: coverage of single best recipe across models: any single label covering most
for ds,ms in ((D,img),(I,img),(S,[m for m in d if S in d[m]])):
    ex=[m for m in ms if True]
    print(ds[-9:],len(ex))
