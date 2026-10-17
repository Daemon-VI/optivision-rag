import json,glob,collections,numpy as np,os
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
fs=sorted(glob.glob(R+'*/selection_rules/*.json'))
rec={}  # (file,rule,target)->{ref:rec}
for f in fs:
    d=json.load(open(f))
    for x in d['results']:
        rec.setdefault((f,x['rule'],x['target']),{})[x['reference']]=x
print(len(fs),len(rec))
rules=sorted(set(k[1] for k in rec)); targets=sorted(set(k[2] for k in rec))
# miss counting
tol=1e-12
for rule in rules:
  for tg in [None]+targets:
    sel=miss=ab=tot=0; lab_sel=lab_miss=lab_ab=0
    for (f,r,t),v in rec.items():
        if r!=rule or (tg is not None and t!=tg): continue
        for ref in ('baseline@1','labels'):
            if ref not in v: continue
            for o in v[ref]['outcomes']:
                tot+=1
                if o['selected'] is None: 
                    if ref=='baseline@1': ab+=1
                    else: lab_ab+=1
                    continue
                m = o['holdout_labels']<t-tol
                if ref=='baseline@1': sel+=1; miss+=m
                else: lab_sel+=1; lab_miss+=m
    print(rule,tg,'b1 sel/miss/abst',sel,miss,ab,'| labels sel/miss/abst',lab_sel,lab_miss,lab_ab)
# check selected values for abstention representation
c=collections.Counter()
for v in rec.values():
    for ref,x in v.items():
        for o in x['outcomes']:
            if o['compression'] is None or o['selected'] is None or o['selected']=='' : c[(ref,str(o['selected'])[:10],o['compression'])]+=1
print(list(c.items())[:10])
