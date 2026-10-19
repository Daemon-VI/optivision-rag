import json,re,numpy as np
src=open('C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/analysis/scripts/corpus_size/pairs.py').read().split("M='ndcg@5'")[0]
exec(src)
M='ndcg@5'
for n,a,b in P:
    A=json.load(open(R+a));B=json.load(open(R+b))
    ra={r['label']:r for r in A['rows']}
    rows=[(r['label'],ra[r['label']],r) for r in B['rows'] if r['label'] in ra]
    print(n,A['n_docs'],B['n_docs'],A['n_queries'],B['n_queries'],len(rows))
    proj=[(l,o,r) for l,o,r in rows if re.search(r'project\((\d+)',l) and int(re.search(r'project\((\d+)',l).group(1))/A['baseline']['dim']<=.25]
    d=[r['retention:'+M]-o['retention:'+M] for l,o,r in proj]
    print(' proj<=25 n',len(d),'mean',np.mean(d),'min',min(d))
    # exclude rows where project appears AFTER a merge? list labels with high CI
    ciw=[(o['retention_ci:'+M][1]-o['retention_ci:'+M][0]) for l,o,r in proj]
    print(' ciw mean',np.mean(ciw))
    # proj rows' retention small
    print(' orig ret range',min(o['retention:'+M] for l,o,r in proj),max(o['retention:'+M] for l,o,r in proj))
    # same on ndcg@10, recall@1
    for m in ['ndcg@10','recall@1']:
        print(' ',m,np.mean([r['retention:'+m]-o['retention:'+m] for l,o,r in proj]))
    # compare labels project dims
    print(' sample',[l for l,_,_ in proj[:6]])
