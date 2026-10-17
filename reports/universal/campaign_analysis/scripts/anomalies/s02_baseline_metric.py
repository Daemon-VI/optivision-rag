# Does measured.json "baseline" (labelled metric ndcg@5) equal ndcg@5 of the wide file? and which metric does row retention follow?
from common import *
from collections import Counter
M=measured(); W=wide_files()
def ds_key(fn):
    return 'BeIR/scifact' if 'scifact' in fn else ('vidore/docvqa_test_subsampled' if 'docvqa' in fn else 'vidore/infovqa_test_subsampled')
res=[];rowcmp=[]
for (folder,fn),w in sorted(W.items()):
    mid=folder.replace('--','/',1); ds=ds_key(fn)
    mm=M[mid][ds]
    b=w['baseline']; bm=mm['baseline']
    which=tuple(k for k in ('ndcg@5','ndcg@10','recall@1','recall@5') if abs(b[k]-bm)<6e-5)
    res.append((mid,ds.split('/')[1][:5],mm['metric'],which))
    # compare retention in measured.json vs wide
    wl={r['label']:r for r in w['rows']}
    n5=n10=tot=0; d5=[];d10=[]
    for r in mm['rows']:
        x=wl.get(r['label'])
        if x is None: continue
        tot+=1
        d5.append(abs(r['retention']-x['retention:ndcg@5'])); d10.append(abs(r['retention']-x['retention:ndcg@10']))
    rowcmp.append((mid,ds.split('/')[1][:5],tot,max(d5) if d5 else None,max(d10) if d10 else None))
print("baseline metric match:",Counter((r[1]=='sciff' or r[1]=='sci', r[3]) for r in res))
for r in res:
    if 'ndcg@5' not in r[3]: print(r)
print("row retention follows (max abs diff vs ndcg@5 / ndcg@10 retention):")
bad=[r for r in rowcmp if r[3] is None or r[3]>1e-4]
print(len(rowcmp),'pairs; ndcg@5-consistent:',len(rowcmp)-len(bad))
for r in bad: print(r)
