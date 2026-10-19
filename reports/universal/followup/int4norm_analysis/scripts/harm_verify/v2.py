import json,glob,numpy as np
from scipy.stats import binomtest
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
B=['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']
res=[]
for b in B:
    for p in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
        res.append((p.replace(chr(92),'/').split('/')[-1],json.load(open(p))))
def row(d,l): return next(r for r in d['rows'] if r['label']==l)
c=0;mn=[]
for n,d in res:
    pq=d['per_query_ndcg@10']
    for a,b in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
        x=np.array(pq[a])-np.array(pq[b]);u=(x>1e-12).sum();dn=(x<-1e-12).sum()
        p=binomtest(int(u),int(u+dn)).pvalue if u+dn else 1
        if x.mean()<0 and p<.05:c+=1;print('SIG',n,a,p)
        if n.startswith('colbertv2.0_nfc') : print(n,a,u,dn)
print('negsig',c)
# EVIE ranks & sign p
for n,d in res:
    if n.startswith('EVIE') and 'tabf' in n:
        pq=d['per_query_ndcg@10']
        for a,b in [('int4norm(mean)','int8(per_vector)'),('int4norm(mean)','int4norm')]:
            x=np.array(pq[a])-np.array(pq[b]);u=(x>1e-12).sum();dn=(x<-1e-12).sum();print(a,b,binomtest(int(u),int(u+dn)).pvalue)
        f=np.array(pq['float32']);
        # ndcg@10 for single relevant = 1/log2(rank+1)
        r=np.where(f>0,np.round(2**(1/np.maximum(f,1e-9))-1),0);print(np.unique(np.round(f,4),return_counts=True))
    # int8 change counts per page unit
    if any(k in n for k in('docvqa','tabf')):
        pq=d['per_query_ndcg@10'];print(n[:40],'int8 chg',int((np.abs(np.array(pq['int8(per_vector)'])-np.array(pq['float32']))>1e-12).sum()),'cos',d['norms'].keys() and '')
# geometry compare
for n,d in res:
    if n.startswith('EVIE'):
        print(n,d['norms']['int4']['ratio_sd'])
# widths
ds={}
for n,d in res:
    k=d['dataset'].split('_')[-1] if False else n.split('_',1)[1]
    for l in ['int8(per_vector)','int4norm','int4norm(mean)']:
        ci=row(d,l)['retention_ci:ndcg@5'];ds.setdefault((k,l),[]).append(ci[1]-ci[0])
for k,v in sorted(ds.items()):print(k,len(v),round(np.median(v),3))
