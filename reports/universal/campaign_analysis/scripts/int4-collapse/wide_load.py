import json,glob,pandas as pd
from optivision.adapters import MODELS
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
recs=[]
reg={k.replace('/','--'):k for k in MODELS}
for f in sorted(glob.glob(R+'campaign/*/wide/*.json'))+sorted(glob.glob(R+'wide/*_test_subsampled.json')):
    d=json.load(open(f)); ff=f.replace(chr(92),'/')
    if 'rows' not in d: continue
    if '/campaign/' in ff:
        fold=ff.split('/campaign/')[1].split('/')[0]; model=reg.get(fold,fold)
    else:
        model=None
    b=d['baseline']; e=b.get('encoder') or {}
    rec=dict(file=ff.replace(R,''),model=model,short=ff.split('/')[-1],dataset=d['dataset'].split('/')[-1],n_docs=d['n_docs'],n_q=d['n_queries'],dim=b['dim'],vpd=b['vectors_per_doc'],base5=b['ndcg@5'],base10=b['ndcg@10'],r1=b['recall@1'],enc_dtype=e.get('dtype'))
    for r in d['rows']:
        if r['group']=='codec':
            for mname in ['ndcg@5','ndcg@10','recall@1','recall@5']:
                rec[f"{r['label']}|{mname}"]=r[f'retention:{mname}']
            rec[r['label']+'|ci']=r.get(f'retention_ci:ndcg@5')
    recs.append(rec)
df=pd.DataFrame(recs); df.to_pickle('wide.pkl')
print(len(df), df.model.isna().sum()); print(df[df.model.isna()].file.tolist())
