import json,glob,os,re
import numpy as np
from scipy import stats
from optivision.adapters import MODELS
mod={k:v.modality for k,v in MODELS.items()}
R=[]
for f in glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json'):
    mid=f.replace(chr(92),'/').split('/campaign/')[1].split('/')[0].replace('--','/',1)
    d=json.load(open(f)); rows=d['rows']
    ret={r['label']:r['retention:ndcg@5'] for r in rows}
    for r in rows:
        if r['group']!='token+codec': continue
        m,c=[x.strip() for x in r['label'].split('>')]
        if m in ret and c in ret:
            ml=1-ret[m]; cl=1-ret[c]; err=r['retention:ndcg@5']-ret[m]*ret[c]
            R.append((mid,d['dataset'],mod.get(mid,'?'),m,c,ml,cl,err,r['vectors_per_doc'],r['dim']))
print(len(R), len({r[0] for r in R}))
import collections
print(collections.Counter((r[2]) for r in R), {mod.get(r[0]) for r in R})
def boot(x,n=2000):
    rng=np.random.default_rng(0);x=np.array(x);return np.percentile([rng.choice(x,len(x)).mean() for _ in range(n)],[2.5,97.5])
def cl_boot(rs,n=2000):
    rng=np.random.default_rng(0);ms=collections.defaultdict(list)
    for r in rs: ms[r[0]].append(r[7])
    k=[np.mean(v) for v in ms.values()]  # model means
    return np.mean(k),np.percentile([np.mean(rng.choice(k,len(k))) for _ in range(n)],[2.5,97.5]),len(k)
for g in ['text','image']:
    rs=[r for r in R if r[2]==g]
    print(g,'all rows mean',np.mean([r[7] for r in rs]),'model-mean',cl_boot(rs))
    for lo,hi in [(-9,.01),(.01,.03),(.03,9)]:
        for cc in [(-9,.01),(-9,9)]:
            s=[r for r in rs if lo<=r[5]<hi and cc[0]<=r[6]<cc[1]]
            if s: print(' merge',lo,hi,'codec<',cc[1],'n',len(s),'models',len({r[0] for r in s}),'rowmean %.4f'%np.mean([r[7] for r in s]),'modelmean',cl_boot(s) if len({r[0] for r in s})>1 else '')
# merge<1 & codec<1 both
for g in ['text','image']:
    s=[r for r in R if r[2]==g and r[5]<.01 and r[6]<.01]
    print(g,'both<1%',len(s),'models',len({r[0] for r in s}))
    per=collections.defaultdict(list)
    for r in s: per[r[0]].append(r[7])
    for k,v in sorted(per.items(),key=lambda x:np.mean(x[1])): 
        if g=='text' or True: pass
    if g=='text':
        for k,v in sorted(per.items(),key=lambda x:np.mean(x[1])): print('  ',k,len(v),'%.4f'%np.mean(v))
# per-model means text all
per=collections.defaultdict(list)
for r in R:
    if r[2]=='text': per[r[0]].append(r[7])
mm={k:np.mean(v) for k,v in per.items()}
for k,v in sorted(mm.items(),key=lambda x:x[1]): print('%-50s %.4f n=%d'%(k,v,len(per[k])))
print('worse than -.04:',sum(v<-.04 for v in mm.values()),len(mm))
# spearman
info={}
for r in R: info[r[0]]=(r[9],stats.__name__)
for g in ['text','image']:
    ms=collections.defaultdict(list);dm={};vm={}
    for r in R:
        if r[2]==g: ms[r[0]].append(r[7]);dm[r[0]]=r[9];vm.setdefault(r[0],[]).append(r[8])
    ks=list(ms); e=[np.mean(ms[k]) for k in ks]
    # baseline vpd: use max vpd
    vp=[max(vm[k]) for k in ks]; dd=[dm[k] for k in ks]
    print(g,len(ks),'dim',stats.spearmanr(dd,e),'vpd',stats.spearmanr(vp,e))
# image only without most extreme, by dataset
for ds in ['docvqa','infovqa']:
    s=[r[7] for r in R if r[2]=='image' and ds in r[1]]
    print(ds,len(s),np.mean(s))
# sensitivity: text drop 3 extreme
t=sorted(mm.values()); print('text median model',np.median(t),'drop worst3 mean',np.mean(t[3:]))
