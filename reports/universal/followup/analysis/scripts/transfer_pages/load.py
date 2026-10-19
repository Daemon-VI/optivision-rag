import json,glob,os
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
def load():
    D={}  # (model,dataset)->{label:row}, meta
    for p in glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'followup/f[45]_*/*/wide/*.json'):
        parts=p.replace(chr(92),'/').split('/')
        mdir=parts[-3]; ds=parts[-1].split('_')[-3] if False else None
        d=json.load(open(p))
        name=parts[-1][:-5]
        for k in ('docvqa','infovqa','arxivqa','tabfquad'):
            if '_'+k+'_' in name: ds=k
        if ds is None: continue
        if 'followup' in p and 'f4' not in p and 'f5' not in p: continue
        D[(mdir,ds)]=dict(rows={r['label']:r for r in d['rows']},nd=d['n_docs'],nq=d['n_queries'],pf=d['pareto_front'],base=d['baseline']['ndcg@5'])
    return D
