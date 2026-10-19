import json,glob,os,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
def s17():
    out={}
    for f in glob.glob(R+'followup/f*b_*/*/int4norm/*.json'):
        f=f.replace(chr(92),'/'); p=f.split('/')
        batch=p[-4]; model=p[-3]; ds=p[-1][:-5].split('_',1)[1]; 
        out[(batch,model,ds)]=f
    return out
def s13():
    out=collections.defaultdict(list)
    for f in glob.glob(R+'followup/*/*/wide/*.json')+glob.glob(R+'campaign/*/wide/*.json'):
        f=f.replace(chr(92),'/'); p=f.split('/')
        model=p[-3]; ds=p[-1][:-5].split('_',1)[1]
        out[(model,ds)].append((p[-4],f))
    return out
def rows(d): return {r['label']:r for r in d['rows']}
