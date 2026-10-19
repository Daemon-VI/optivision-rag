import json,glob,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
a=json.load(open(R+'f5_pages_wide/tencent--EVIE-4.5B/wide/EVIE-4.5B_tabfquad_test_subsampled.json'))
b=json.load(open(glob.glob(R+'f11b*/*EVIE*/int4norm/*tabf*.json')[0]))
ra={r['label']:r for r in a['rows']}
for r in b['rows']:
    if r['label'] in ra:
        o=ra[r['label']]
        print(r['label'],[round(r[k]-o[k],10) for k in ('ndcg@5','ndcg@10','retention:ndcg@5')], r['retention_ci:ndcg@5']==o['retention_ci:ndcg@5'])
print(ra['int4(mean)']['retention:ndcg@5'],ra['int4(mean)']['retention_ci:ndcg@5'])
# flip sim Q6 colpali: observed @5 diff + bootstrap deviation of @10 diff
U={}
for p in glob.glob(R+'f10b*/*colpali-v1.3/int4norm/*docvqa*.json')+glob.glob(R+'f11b*/*colpali-v1.3/int4norm/*docvqa*.json'):
    d=json.load(open(p));pq=d['per_query_ndcg@10']
    f=np.array(pq['float32']);i=np.array(pq['int4']);n=np.array(pq['int4norm'])
    rng=np.random.default_rng(1);N=len(f);idx=rng.integers(0,N,(4000,N))
    dev=((n[idx].mean(1)/f[idx].mean(1))-(i[idx].mean(1)/f[idx].mean(1)))
    dev=dev-dev.mean()
    obs=[r for r in d['rows'] if r['label']=='int4norm'][0]['retention:ndcg@5']-[r for r in d['rows'] if r['label']=='int4'][0]['retention:ndcg@5']
    print(p[-60:],obs,'P(diff<-0.01)',((obs+dev)<-0.01).mean())
