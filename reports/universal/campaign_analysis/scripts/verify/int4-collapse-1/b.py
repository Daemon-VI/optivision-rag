import json
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
from optivision.adapters import MODELS
rows=[];c=[]
for m,ds in M.items():
    for d,v in ds.items():
        R={r['label']:r for r in v['rows']}
        i=R.get('int4');k=R.get('int4(mean)')
        if i is None or k is None: print('missing',m,d,[l for l in ('int4','int4(mean)') if l not in R]);continue
        e=MODELS[m].evidence if m in MODELS else ''
        rows.append((m,d[6:12],i['retention'],i['retention_interval'][1],k['retention'],k['retention_interval'][0],e))
print(len(rows))
print("int4 sorted:")
for r in sorted(rows,key=lambda r:r[2])[:20]: print(r[0],r[1],round(r[2],3),round(r[3],3),'cent',round(r[4],3),r[6][:60])
print("n int4<0.9",sum(r[2]<0.9 for r in rows),"n<0.98",sum(r[2]<0.98 for r in rows))
low=[r for r in rows if r[4]<0.98]
print("centred<0.98:",[(r[0],r[1],round(r[4],4)) for r in low], "of",len(rows))
print("centred<0.99:",[(r[0],r[1],round(r[4],4)) for r in rows if r[4]<0.99])
for m in MODELS:
    if 'nemotron' in m or 'EVIE' in m: print(m,MODELS[m].evidence,MODELS[m].dimension)
