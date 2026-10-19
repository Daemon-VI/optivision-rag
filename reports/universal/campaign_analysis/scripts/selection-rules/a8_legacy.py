import sys; sys.path.insert(0,'.')
from load import *
import glob,pathlib
U='C:/Users/Lenovo/projects/optivision-package/reports/universal'
def lf(f):
    d=json.load(open(f)); rows=[]
    for r in d['results']:
        if r['reference']!='labels': continue
        for o in r['outcomes']:
            if o['selected']: rows.append(dict(rule=r['rule'],target=r['target'],seed=o['seed'],sel=o['selected'],hold=o['holdout'],comp=o['compression']))
            else: rows.append(dict(rule=r['rule'],target=r['target'],seed=o['seed'],sel=None))
    return d,pd.DataFrame(rows)
leg=sorted(glob.glob(U+'/selection_rules/*.json'))
print('legacy files',[pathlib.Path(f).name for f in leg]); 
print('n_candidates', {pathlib.Path(f).name:json.load(open(f))['n_candidates'] for f in leg})
# compare overlap: E2-colpali-docvqa vs campaign vidore/colpali-v1.2 docvqa; others by name
pairs=[('selection_rules/E2-colpali-docvqa.json','campaign/vidore--colpali-v1.2/selection_rules/colpali-v1.2_docvqa_test_subsampled.json'),
('selection_rules/E2-colpali-infovqa.json','campaign/vidore--colpali-v1.2/selection_rules/colpali-v1.2_infovqa_test_subsampled.json')]
for a,b in pairs:
    try:
        da,A=lf(U+'/'+a); db,B=lf(U+'/'+b)
        m=A.merge(B,on=['rule','target','seed'],suffixes=('_old','_new'))
        print(a,'vs',b,'rows',len(m),'same choice',(m.sel_old==m.sel_new).mean().round(3),'nq',da['n_queries'],db['n_queries'],'cands',da['n_candidates'],db['n_candidates'])
        print(m.groupby('rule').apply(lambda g:(g.sel_old==g.sel_new).mean()).round(2).to_dict())
    except Exception as e: print('ERR',a,b,e)
# which legacy models are in campaign by name
for f in leg:
    n=pathlib.Path(f).name; print(n)
print(len(glob.glob(U+'/campaign/batches/*/orphans/*/selection_rules/*.json')),glob.glob(U+'/campaign/batches/*/orphans/*/selection_rules/*.json'))
print(open(U+'/campaign/README.md',encoding='utf8').read()[:3000])
