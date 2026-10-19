import json,glob,os,numpy as np
from scipy import stats
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
rows=[]
for f in sorted(glob.glob(R+'geometry/*.json')):
    g=json.load(open(f)); n=os.path.basename(f)[:-5]
    fr=json.load(open(R+'frontier/'+n+'.json'))
    d={r['label']:r for r in fr['rows']}
    gm=g['metric']
    m='ndcg@5'
    def ret(l,m): return d[l]['retention:'+m]
    rows.append(dict(n=n,qc=g['query_cosine_to_corpus_mean_median'],mn=g['corpus_mean_norm'],nn=g['median_nn_cosine_within_doc'],
      i5=ret('none > int4','ndcg@5'),i10=ret('none > int4','ndcg@10'),b5=ret('none > binary','ndcg@5'),b10=ret('none > binary','ndcg@10'),
      cb=g['rows']['binary centred']['retention'],gm=gm,fm=fr['frontier_metric'],ci=d['none > int4'].get('ci:ndcg@5')))
for r in rows: print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
def sp(x,y,idx):
    x=[x[i] for i in idx];y=[y[i] for i in idx]
    r=stats.spearmanr(x,y);return round(r.statistic,3),round(r.pvalue,3)
for key in ['i5','i10']:
  for pred in ['qc','mn','nn']:
    a=[r[pred] for r in rows];b=[r[key] for r in rows]
    print(key,pred,'all7',sp(a,b,range(7)),'img6',sp(a,b,range(6)))
# leave-one-out, image only collapse of ties
a=[r['qc'] for r in rows];b=[r['i5'] for r in rows]
print('ties in i5 image', sorted(b[:6]))
for k in range(7):
    idx=[i for i in range(7) if i!=k]; print('drop',rows[k]['n'],sp(a,b,idx))
# permutation: exact p for text-only driver; without colpali docvqa among images
idx=[i for i in range(6) if rows[i]['n']!='E2-colpali-docvqa']
print('img w/o colpali-docvqa',sp(a,b,idx))
# with text, sibling-collapsed (per-model mean)
