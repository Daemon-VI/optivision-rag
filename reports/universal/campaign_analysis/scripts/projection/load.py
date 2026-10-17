import json,glob,os,re
import pandas as pd
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
files=glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'wide/*.json')+glob.glob(R+'text/text-*-scifact.json')
rows=[];base=[]
for f in files:
    if os.path.basename(f).startswith('smoke'): continue
    try: d=json.load(open(f))
    except Exception as e: print('ERR',f,e); continue
    if 'rows' not in d or 'baseline' not in d: print('skip',f); continue
    enc=d['baseline'].get('encoder',{}); model=enc.get('model') or os.path.basename(f)
    ds=d['dataset']; b=d['baseline']
    base.append(dict(model=model,ds=ds,file=f,D=b['dim'],vpd=b['vectors_per_doc'],base=b['ndcg@5'],dtype=enc.get('dtype'),nq=d['n_queries'],nd=d['n_docs']))
    for r in d['rows']:
        st=r['pipeline']['stages']
        rows.append(dict(model=model,ds=ds,D=b['dim'],group=r['group'],label=r['label'],stages=tuple((s['stage'],s.get('method'),s.get('dim')) for s in st),
          nst=len(st),dim=r['dim'],ret=r['retention:ndcg@5'],lo=r['retention_ci:ndcg@5'][0],hi=r['retention_ci:ndcg@5'][1],
          r10=r.get('retention:ndcg@10'),r1=r.get('retention:recall@1'),nd5=r['ndcg@5'],base=b['ndcg@5'],comp=r['compression_vs_float32'],vpd=r['vectors_per_doc']))
df=pd.DataFrame(rows);bs=pd.DataFrame(base)
if __name__=='__main__':
    print(len(files),len(bs),bs.model.nunique(),bs.ds.value_counts().to_dict())
    print(bs.groupby('model').D.first().value_counts().sort_index())
    p=df[df.stages.apply(lambda s:len(s)==1 and s[0][0]=='project')]
    print(len(p)); print(p.groupby(['dim']).size().to_dict())
    print(p.label.value_counts().head(40))
    print(bs[bs.duplicated(['model','ds'],keep=False)])
