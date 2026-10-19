import json,numpy as np,collections
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
S={'sci':'BeIR/scifact','doc':'vidore/docvqa_test_subsampled','info':'vidore/infovqa_test_subsampled'}
bins=[(16,64),(64,256),(256,1e9)]
rows=[]
for m,ds in d.items():
    for k,v in S.items():
        if v in ds:
            for r in ds[v]['rows']:
                rows.append((m,k,r['label'],r['compression'],r['retention'],r['retention_interval'][0],r['vectors_per_doc']))
print(len(rows),collections.Counter(r[1] for r in rows),len({r[0] for r in rows if r[1]=='sci'}))
rng=np.random.default_rng(0)
def boot(vals_by_model,n=4000):
    ms=list(vals_by_model);a=[]
    for _ in range(n):
        s=rng.integers(0,len(ms),len(ms));a.append(np.mean([vals_by_model[ms[i]] for i in s]))
    return np.percentile(a,[2.5,97.5])
for lo,hi in bins:
    out=[]
    for k in S:
        sel=[r for r in rows if r[1]==k and lo<=r[3]<hi]
        pm=collections.defaultdict(list)
        for r in sel: pm[r[0]].append(r[4])
        mm={m:np.mean(v) for m,v in pm.items()}
        ci=boot(mm)
        out.append(f"{k}: rowmean {np.mean([r[4] for r in sel]):.3f} n={len(sel)} models={len(mm)} modelmean {np.mean(list(mm.values())):.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] lb>=.95 {np.mean([r[5]>=.95 for r in sel]):.3f}")
    print((lo,hi));print('\n'.join(out))
# matched pipeline labels
lab={k:collections.defaultdict(dict) for k in S}
for r in rows: lab[r[1]][r[2]][r[0]]=r[4]
common=set(lab['sci'])&set(lab['doc'])&set(lab['info'])
print('common labels',len(common))
# compression per label by dataset (median)
comp=collections.defaultdict(lambda:collections.defaultdict(list))
for r in rows: comp[r[2]][r[1]].append(r[3])
res=[]
for l in sorted(common):
    s=np.mean(list(lab['sci'][l].values()));dd=np.mean(list(lab['doc'][l].values()));ii=np.mean(list(lab['info'][l].values()))
    res.append((l,s,dd,ii,{k:round(float(np.median(v)),1) for k,v in comp[l].items()}))
res.sort(key=lambda x:x[4]['doc'])
for x in res: print(x[0][:40].ljust(40),f"{x[1]:.3f} {x[2]:.3f} {x[3]:.3f}",x[4])
sd=np.array([x[1]-x[2] for x in res]);si=np.array([x[1]-x[3] for x in res])
print('labels sci<doc',(sd<0).sum(),'sci<info',(si<0).sum(),len(res),'mean diff',sd.mean(),si.mean())
# per model min over common labels
for k in S:
    mn={}
    for m in {r[0] for r in rows if r[1]==k}:
        v=[lab[k][l][m] for l in common if m in lab[k][l]]
        mn[m]=(round(min(v),3),len(v))
    print(k,sorted(mn.items(),key=lambda x:x[1])[:20] if k=='sci' else (np.median([x[0] for x in mn.values()]),min(mn.values()),max(mn.values())))
