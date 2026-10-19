import json,glob
for f in sorted(glob.glob('followup/f[1-3]*/*/wide/*.json')+glob.glob('campaign/*/wide/*.json')):
    d=json.load(open(f))
    if 'rows' not in d: continue
    R={r['label']:r for r in d['rows']}
    if 'int4' not in R: continue
    p=f.replace(chr(92),'/').split('/')
    n=p[-3]
    if any(k in n.lower() for k in ['answerai','lateon','mxbai-edge','gte-modern','lfm2']):
        r=R['int4']
        print(p[-4],n,p[-1],d['n_queries'],d['n_docs'],round(r['retention:ndcg@10'],3),[round(x,3) for x in r['retention_ci:ndcg@10']], round(R['int4(mean)']['retention:ndcg@10'],3) if 'int4(mean)' in R else None)
