import json,glob,statistics as st,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
fams={'text':['f9b_int4norm_text'],'pages':['f10b_int4norm_pages_128d','f11b_int4norm_pages_wide'],'sci':['f12b_text_scidocs']}
K='retention:ndcg@5';C='retention_ci:ndcg@5'
H='hierarchical_merge'
def load(f):
    return [json.load(open(p)) for b in fams[f] for p in sorted(glob.glob(R+b+'/*/int4norm/*.json'))]
def row(d,l):
    for r in d['rows']:
        if r['label']==l: return r
for f in fams:
    U=load(f);print('==',f,len(U),sorted(set(d['n_queries'] for d in U)))
    labs=[r['label'] for r in U[0]['rows']]
    for l in ['int4norm(mean)','int8(per_vector)','int4(mean)','lloyd2(7)','binary',H+'(0.33) > int4norm(mean)',H+'(0.25) > int4norm(mean)']:
        v=[row(d,l)[K] for d in U]
        print(l,sum(x>=.95 for x in v),sum(row(d,l)[C][0]>=.95 for d in U),round(st.mean(v),4))
    def pair(a,b):
        A=[row(d,a) for d in U];B=[row(d,b) for d in U]
        diff=[x[K]-y[K] for x,y in zip(A,B)]
        print(a[:35],'-',b[:35],round(st.mean(diff),4),sum(x>1e-9 for x in diff),sum(x<-1e-9 for x in diff),'sepA',sum(x[C][0]>y[C][1] for x,y in zip(A,B)),'sepB',sum(y[C][0]>x[C][1] for x,y in zip(A,B)),'bytes',round(st.mean(x['bytes_per_doc']/y['bytes_per_doc'] for x,y in zip(A,B)),3))
    for a,b in [('int4norm(mean)','int8(per_vector)'),('int4norm(mean)','int4(mean)'),(H+'(0.33) > int4norm(mean)',H+'(0.33) > int4(mean)'),(H+'(0.25) > int4norm(mean)',H+'(0.25) > int4(mean)'),(H+'(0.33) > int4norm(mean)',H+'(0.33) > int8(per_vector)'),(H+'(0.33) > int8(per_vector)','int4norm(mean)'),('int4norm(mean)','lloyd2(7)'),('lloyd2(7)',H+'(0.33) > int4norm(mean)'),('binary','int4norm(mean)'),(H+'(0.33) > int4norm(mean)','int4norm(mean)')]:
        pair(a,b)
