import json,glob,re,os,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
files=glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'wide/*.json')+glob.glob(R+'wide/*/*.json')
U={}
for f in files:
    f=f.replace(chr(92),'/')
    try: d=json.load(open(f))
    except Exception: continue
    if 'rows' not in d or 'baseline' not in d: continue
    parts=f.split('/')
    ds='docvqa' if 'docvqa' in d['dataset'] else ('infovqa' if 'infovqa' in d['dataset'] else d['dataset'])
    if 'campaign' in parts: model=parts[parts.index('campaign')+1]
    else: model=os.path.basename(f).split('_'+ds)[0]
    U.setdefault((model,ds),[]).append((f,d))
M='retention:ndcg@5'
def rows(d): return {r['label']:r for r in d['rows']}
