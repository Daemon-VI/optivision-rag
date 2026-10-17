import json,glob
for f in glob.glob("C:/Users/Lenovo/projects/optivision-package/reports/universal/**/*.json",recursive=True):
    if not any(k in f.lower() for k in ('nemotron','evie','lateon')) or 'selection' in f or 'run.json' in f or 'outcomes' in f: continue
    try: d=json.load(open(f))
    except: continue
    if not isinstance(d,dict) or 'rows' not in d: continue
    R={r['label']:r for r in d['rows']}
    out=[]
    for l in ('float16','int8(per_vector)','int4','int4(mean)','binary','lloyd2(7)'):
        if l in R: r=R[l]; out.append((l,round(r['retention:ndcg@5'],3),[round(x,3) for x in r['retention_ci:ndcg@5']]))
    print(f.replace(chr(92),'/').split('universal/')[1], d['baseline'].get('ndcg@5'));
    for o in out: print('   ',o)
