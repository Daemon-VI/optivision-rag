import json,glob,os,numpy as np,collections
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/'
M='retention:ndcg@5'
def kind(p):
    s=p['stages']
    return tuple((x['stage'],x.get('center'),x.get('scale')) for x in s)
CODEC={'binary':('binary',None,None),'int4':('int4',None,None),'lloyd2':('lloyd2',None,None),'int4(mean)':('int4','mean',None),'int8':('int8',None,'per_vector')}
recs=[]  # model,dataset,ratio,codec,actual,prod
files=glob.glob(R+'*/wide/*.json')
print(len(files))
for f in files:
    d=json.load(open(f)); model=f.replace(chr(92),'/').split('/campaign/')[1].split('/')[0]
    ds=d['dataset']
    rows={}
    for r in d['rows']:
        rows[kind(r['pipeline'])+(('r',r['pipeline']['stages'][0].get('ratio')) if r['pipeline']['stages'] and r['pipeline']['stages'][0]['stage']=='hierarchical_merge' else ('x',),)]=r
    cod={}
    for cn,k in CODEC.items():
        key=(k,('x',))
        if key in rows: cod[cn]=rows[key][M]
    for r in d['rows']:
        st=r['pipeline']['stages']
        if len(st)==2 and st[0]['stage']=='hierarchical_merge':
            mk=((st[0]['stage'],None,None),('r',st[0]['ratio']))
            if mk not in rows: continue
            m=rows[mk][M]
            k2=kind({'stages':st[1:]})[0]
            for cn,k in CODEC.items():
                if k==k2 and cn in cod:
                    recs.append((model,ds,st[0]['ratio'],cn,r[M],m*cod[cn],m,cod[cn]))
print(len(recs))
import pickle;pickle.dump(recs,open('recs.pkl','wb'))
text=lambda ds:'scifact' in ds.lower()
rng=np.random.default_rng(0)
def boot(vals,n=10000):
    v=np.array(vals);idx=rng.integers(0,len(v),(n,len(v)));m=v[idx].mean(1);return np.percentile(m,[2.5,97.5])
res={}
for cn in CODEC:
  for grp in ('text','image'):
    per=collections.defaultdict(list)
    for (mo,ds,ra,c,a,p,m,cc) in recs:
        if c==cn and text(ds)==(grp=='text'): per[mo].append(a-p)
    mm={k:np.mean(v) for k,v in per.items()}
    v=np.array(list(mm.values()))
    if len(v)==0: continue
    lo,hi=boot(v)
    res[(cn,grp)]=mm
    print(cn,grp,len(v),'mean %.4f [%.4f,%.4f]'%(v.mean(),lo,hi),'median %.4f'%np.median(v),'min %.4f max %.4f'%(v.min(),v.max()))
# sorted text models binary
for cn in ('binary','int4','lloyd2'):
    print(cn,sorted((round(x,3),k.split('--')[-1]) for k,x in res[(cn,'text')].items()))
