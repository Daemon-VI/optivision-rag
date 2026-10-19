import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy import stats
rng=np.random.default_rng(2)
U=load()
def boot(x,B=10000):
    idx=rng.integers(0,len(x),(B,len(x))); return np.percentile(x[idx].mean(1),[2.5,97.5])
def rank_from_ndcg(v):
    # single relevant doc: ndcg@10 = 1/log2(rank+1); 0 if beyond 10
    out=[]
    for x in v:
        out.append(0 if x<1e-9 else int(round(2**(1/x)-1)))
    return np.array(out)
for key in [('EVIE-4.5B','tabfquad_test_subsampled'),('nemotron-colembed-vl-4b-v2','tabfquad_test_subsampled'),('EVIE-4.5B','docvqa_test_subsampled')]:
    pq=U[key]['d']['per_query_ndcg@10']; f=np.array(pq['float32'])
    r=rank_from_ndcg(f); print(key,'float32 rank of relevant: counts rank 1..6,7+,none',[int((r==i).sum()) for i in range(1,7)],int((r>=7).sum()),int((r==0).sum()),'distinct ndcg vals',len(set(np.round(f,4))))
key=('EVIE-4.5B','tabfquad_test_subsampled'); d=U[key]['d']; pq=d['per_query_ndcg@10']
f=np.array(pq['float32'])
print('\nEVIE tab paired differences in ndcg@10 (mean, boot CI, changed up/down, sign p)')
for a,b in [('int8(per_vector)','float32'),('int4','float32'),('int4norm','float32'),('int4(mean)','float32'),('int4norm(mean)','float32'),('int4(mean)','int4'),('int4norm(mean)','int4norm'),('int4norm(mean)','int8(per_vector)')]:
    x=np.array(pq[a])-np.array(pq[b]); up=(x>1e-9).sum(); dn=(x<-1e-9).sum()
    print('%-18s - %-17s %+.4f [%+.4f,%+.4f] up %3d down %3d signP %.3f'%(a,b,x.mean(),*boot(x),up,dn,stats.binomtest(int(up),int(up+dn),.5).pvalue))
# rank transitions float32 -> int4norm(mean)
ra=rank_from_ndcg(np.array(pq['int4norm(mean)'])); r=rank_from_ndcg(f); r8=rank_from_ndcg(np.array(pq['int8(per_vector)']))
import collections
print('\nrank shift of relevant doc, int4norm(mean) vs float32: mean abs change among changed %.2f ; moved from rank1 to >1: %d ; moved into rank1: %d'%(np.abs(ra-r)[ra!=r].mean(),((r==1)&(ra!=1)).sum(),((r!=1)&(ra==1)).sum()))
print('int8: moved from rank1 to >1: %d ; into rank1: %d'%(((r==1)&(r8!=1)).sum(),((r!=1)&(r8==1)).sum()))
print('rank1 count: float32 %d int8 %d int4 %d int4norm %d int4(mean) %d int4norm(mean) %d'%tuple(((rank_from_ndcg(np.array(pq[l]))==1).sum()) for l in ['float32','int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']))
print('recall@1 values in rows:',{l:round(row(d,l)['recall@1'],3) for l in ['float32','int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']})
# qrels
q=json.load(open(R+'f11b_int4norm_pages_wide/tencent--EVIE-4.5B/EVIE-4.5B_tabfquad_test_subsampled_qrels.json'))
print('qrels type',type(q).__name__, 'n',len(q))
vals=list(q.values()) if isinstance(q,dict) else q
print('relevant docs per query dist',collections.Counter(len(v) if hasattr(v,'__len__') else 1 for v in vals).most_common(4))
# distribution of bootstrap-sd by n: for the same codec compare half-widths of float32-relative CI across page tab units
print('\nint4norm(mean) @5 retention CI half-width, tabfquad units:')
for k,u in U.items():
    if k[1].startswith('tabf'):
        ci=row(u['d'],'int4norm(mean)')['retention_ci:ndcg@5']; print('  %-36s pt %.3f ci [%.3f,%.3f] halfwidth %.3f'%(k[0],row(u['d'],'int4norm(mean)')['retention:ndcg@5'],ci[0],ci[1],(ci[1]-ci[0])/2))
# cross-run reproducibility: s13 (f5) vs s17 rows for EVIE tab
w=json.load(open(R+'f5_pages_wide/tencent--EVIE-4.5B/wide/EVIE-4.5B_tabfquad_test_subsampled.json'))
print('\nreproducibility s13 vs s17 (ndcg@5 / @10 exact diff):')
for l in ['float32','float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary']:
    a,b=row(w,l),row(d,l); print('  %-18s %.6f %.6f diff %.2e | @10 diff %.2e'%(l,a['ndcg@5'],b['ndcg@5'],a['ndcg@5']-b['ndcg@5'],a['ndcg@10']-b['ndcg@10']))
