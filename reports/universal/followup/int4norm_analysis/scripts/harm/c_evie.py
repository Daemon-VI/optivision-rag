import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
U=load()
w=json.load(open(R+'f5_pages_wide/tencent--EVIE-4.5B/wide/EVIE-4.5B_tabfquad_test_subsampled.json'))
print('f5 wide keys',list(w), 'n_docs',w['n_docs'],'nq',w['n_queries'], 'baseline',w['baseline'])
print('wide row keys',list(w['rows'][0]))
for l in ['float32','float16','int8(per_vector)','int4','int4(mean)']:
    r=row(w,l); print('f5 s13',l,round(r['ndcg@5'],4),round(r['retention:ndcg@5'],4),r.get('retention_ci:ndcg@5'))
d=U[('EVIE-4.5B','tabfquad_test_subsampled')]['d']
print('s17 baseline',d['baseline']['ndcg@5'],d['baseline']['ndcg@10'],'n_docs',d['n_docs'])
for l in ['float32','float16','int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)','lloyd2(7)','binary']:
    r=row(d,l); print('s17',l,round(r['ndcg@5'],4),round(r['retention:ndcg@5'],4),[round(x,3) for x in r['retention_ci:ndcg@5']],'| @10',round(r['retention:ndcg@10'],4),[round(x,3) for x in r['retention_ci:ndcg@10']])
print('baseline s13 vs s17 ndcg@5:',w['baseline']['ndcg@5'],d['baseline']['ndcg@5'])
dd=U[('EVIE-4.5B','docvqa_test_subsampled')]['d']
for nm,x in [('tab',d),('docvqa',dd)]:
    print(nm,'norms:',json.dumps({k:(v if not isinstance(v,dict) else {a:round(b,4) for a,b in v.items()}) for k,v in x['norms'].items()}))
# geometry
for ds in ['tabfquad_test_subsampled','docvqa_test_subsampled']:
    g=json.load(open(R+'f11b_int4norm_pages_wide/tencent--EVIE-4.5B/geometry/EVIE-4.5B_%s.json'%ds))
    print(ds,{k:(round(v,4) if isinstance(v,float) else v) for k,v in g.items() if k not in('codecs',)})
    print(' codecs',json.dumps(g['codecs'])[:700])
g5=json.load(open(R+'f5_pages_wide/tencent--EVIE-4.5B/geometry/EVIE-4.5B_tabfquad_test_subsampled.json'))
print('f5 geometry tab', {k:(round(v,4) if isinstance(v,float) else v) for k,v in g5.items() if k!='codecs'})
# per-query structure
pq=d['per_query_ndcg@10']; f=np.array(pq['float32'])
print('\nEVIE tab per-query @10: float32 mean %.4f; n zero %d; n==1 %d'%(f.mean(),(f==0).sum(),(f==1).sum()))
for l in ['int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']:
    x=np.array(pq[l])-f
    big=np.abs(x)>0.3
    print('%-16s changed %3d up %3d down %3d  |chg|>0.3: %3d  sum_down %.2f sum_up %.2f   largest drop %.2f'%(l,(abs(x)>1e-9).sum(),(x>1e-9).sum(),(x<-1e-9).sum(),big.sum(),x[x<0].sum(),x[x>0].sum(),x.min()))
# overlap of queries changed between codecs
ch={l:set(np.where(abs(np.array(pq[l])-f)>1e-9)[0]) for l in ['int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']}
print('overlap int8 & int4(mean):',len(ch['int8(per_vector)']&ch['int4(mean)']),'int4(mean)&int4norm(mean):',len(ch['int4(mean)']&ch['int4norm(mean)']),'of',len(ch['int4(mean)']),len(ch['int4norm(mean)']))
# ties? baseline ranks heavily tied? check number of queries with fractional ndcg
print('distinct float32 per-query values',len(set(np.round(f,6))))
# query vector info
qv=d['query_vectors']; print('query_vectors sample',type(qv),qv[:5] if isinstance(qv,list) else '')
dq=dd['query_vectors']; print('docvqa qv',dq[:5])
print('tab mean qvecs',np.mean(qv),'docvqa',np.mean(dq))
print(d['environment'])
