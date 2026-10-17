import json,numpy as np
from scipy import stats
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
labs=['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)']
rng=np.random.default_rng(0)
ms=[m for m,v in d.items() if len(v)>=2]
dsn=sorted(d[ms[0]])
print(dsn)
def get(m,ds,l):
    for r in d[m][ds]['rows']:
        if r['label']==l: return r['retention']
def run(ms,tag):
    print(tag,len(ms))
    for l in labs:
        a=[];b=[];n=[]
        for m in ms:
            x=get(m,dsn[0],l);y=get(m,dsn[1],l)
            if x is None or y is None: continue
            a.append(1-x);b.append(1-y);n.append(m)
        a=np.array(a);b=np.array(b);df=a-b
        bs=[rng.choice(df,len(df)).mean() for _ in range(5000)]
        try:w=stats.wilcoxon(df).pvalue
        except: w=None
        print(l,len(df),'ratio %.2f'%(a.mean()/b.mean()),'doc %.4f info %.4f'%(a.mean(),b.mean()),'diff %.2fpp CI %.2f..%.2f'%(df.mean()*100,np.percentile(bs,2.5)*100,np.percentile(bs,97.5)*100),'W p=%.4f'%w,'docworse',(df>0).sum(),'sign p=%.3f'%stats.binomtest(int((df>0).sum()),int((df!=0).sum())).pvalue,'median %.2fpp'%(np.median(df)*100))
        if tag=='all' and l.endswith('(0.1)') or tag=='all' and l.startswith('adap'):
            o=np.argsort(-np.abs(df))[:3];print('  top',[(n[i][:25],round(a[i],3),round(b[i],3)) for i in o])
run(ms,'all')
# drop top model by doc loss, and sibling families
def fam(m): 
    return m.split('--')[0]+'|'+''.join(c for c in m.split('--')[1].lower() if c.isalpha())[:6]
seen=set();uniq=[]
for m in ms:
    f=m.split('--')[1].lower().replace('-','')[:7]
    if f in seen: continue
    seen.add(f);uniq.append(m)
run(uniq,'dedupe-prefix')
# float16 baselines excluded
f16=[m for m in ms if 'float16' in str(d[m][dsn[0]].get('note',''))+str(d[m][dsn[1]].get('note',''))]
print('f16 notes',len(f16), [m for m in ms][:0])
run([m for m in ms if m not in f16],'no-f16')
