import json,glob,os,collections,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/'
C=R+'reports/universal/campaign/'
M=json.load(open(R+'src/optivision/presets/measured.json'))['models']
print(list(M['LiquidAI/LFM2-ColBERT-350M'].keys()), list(M['vidore/colpali'].keys()))
wide=[]
for f in glob.glob(C+'*/wide/*.json'):
    f=f.replace(chr(92),'/'); d=json.load(open(f)); mid=f.split('/')[-3].replace('--','/',1); wide.append((mid,d))
cnt=collections.Counter(); mis=collections.Counter(); ex=[]
for mid,d in wide:
    ds=d['dataset']; key=[k for k in M.get(mid,{}) if any(t in k and t in ds for t in ('docvqa','infovqa','scifact'))]
    if len(key)!=1: print('nokey',mid,ds,key); continue
    v=M[mid][key[0]]; b=d['baseline']
    txt='scifact' in ds
    cnt[(txt,abs(v['baseline']-b['ndcg@5'])<6e-5,abs(v['baseline']-b['ndcg@10'])<6e-5)]+=1
    lab={r['label']:r for r in d['rows']}
    for r in v['rows']:
        w=lab.get(r['label'])
        if not w: mis[(txt,'nolabel')]+=1;continue
        mis[(txt,abs(r['retention']-w['retention:ndcg@5'])<6e-5,abs(r['retention']-w['retention:ndcg@10'])<6e-5)]+=1
print(cnt);print(mis)

# precision
runs={}
for f in glob.glob(C+'*/run.json'):
    d=json.load(open(f.replace(chr(92),'/'))); runs[d['model']]=d
tab=collections.Counter()
for mid,r in runs.items():
    cr=r.get('check_retriever') or {}; u=cr.get('unit_norm_max_error'); dt=(r.get('spec') or {}).get('dtype')
    if u is None: tab[(r['kind'],dt,'none')]+=1; continue
    tab[(r['kind'],dt,'fp32like' if u<2e-7 else ('fp16like' if u<1e-3 else 'big'))]+=1
for k,v in sorted(tab.items(),key=str): print(k,v)
# compression
bad=0;n=0;mx=0
for mid,d in wide:
    for r in d['rows']:
        n+=1
        c=d['baseline']['float32_bytes_per_doc']/r['bytes_per_doc']
        mx=max(mx,abs(c/r['compression_vs_float32']-1))
        # bytes from vectors x dim x bits
print('rows',n,'maxrel compression err',mx)
# float16 baseline models: dtype float16 count
print('fp16 spec', sum(1 for r in runs.values() if (r.get('spec') or {}).get('dtype')=='float16'))
# merge
for mid,d in wide:
    if 'olSmol-500M' in mid or 'colsmolvlm' in mid:
        for r in d['rows']:
            if r['label'].startswith('hierarchical_merge(0.1)') or r['label']=='hierarchical_merge(0.1)':
                print(mid,d['dataset'][-20:],r['label'],r['vectors_per_doc']/d['baseline']['vectors_per_doc'],r['compression_vs_float32'])
# other models merge
rat=[]
for mid,d in wide:
    for r in d['rows']:
        if r['label']=='hierarchical_merge(0.1)': rat.append((mid.split('/')[-1],round(r['vectors_per_doc']/d['baseline']['vectors_per_doc'],3)))
print(sorted(rat,key=lambda x:x[1])[:60])
# row counts
cs=collections.Counter(len(d['rows']) for _,d in wide);print(cs)
labs=[set(r['label'] for r in d['rows']) for _,d in wide]
print(len(set.union(*labs)),len(set.intersection(*labs)))
