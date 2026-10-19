import json,glob
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
for p in glob.glob(R+'f*b_*/*/int4norm/*.json'):
    n=p.replace(chr(92),'/').split('/')[-1]
    if n.startswith(('LateOn_','mLateOn','mxbai-edge-colbert-v0-17m_nf','nemotron-colembed-vl-4b-v2_tab')) :
        d=json.load(open(p));r={x['label']:x for x in d['rows']}
        print(n,round(r['int4norm']['retention:ndcg@5'],4),r['int4norm']['retention_ci:ndcg@5'][0].__round__(3))
