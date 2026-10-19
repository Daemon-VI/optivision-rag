import json,numpy as np
from scipy.stats import spearmanr
B='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/analysis/scripts/int4_answerai/'
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
cfg={'nfcorpus':R+'f1_text_nfcorpus_a/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_nfcorpus_qrels.json','scifact':R+'f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_scifact_qrels.json'}
def ndcg(top,rel):
    g=sum(1/np.log2(i+2) for i,d in enumerate(top) if d in rel); return g/sum(1/np.log2(i+2) for i in range(min(10,len(rel))))
def enc(v,mode):
    a=np.maximum(np.abs(v).max(1),1e-4).astype(np.float16).astype(np.float32)
    q=np.clip(np.round(v/a[:,None]*7),-7,7)
    d=q*(a[:,None]/7)
    if mode=='renorm': d=d/np.linalg.norm(d,axis=1,keepdims=True)*np.linalg.norm(v,axis=1,keepdims=True)
    return d.astype(np.float32)
rng=np.random.default_rng(1)
for ds,qrf in cfg.items():
    z=np.load(B+f'vec/{ds}_docs.npz'); D=z['D']; lens=z['lens']; ids=[str(i) for i in z['ids']]; pos={d:i for i,d in enumerate(ids)}
    zq=np.load(B+f'vec/{ds}_queries.npz'); Q=zq['Q']; qids=[str(q) for q in zq['qids']]; qr=json.load(open(qrf)); st=np.concatenate([[0],np.cumsum(lens)[:-1]])
    print(ds,'unit-norm fp docs?',float(np.linalg.norm(D,axis=1).mean()),float(np.linalg.norm(D,axis=1).std()),'keys',list(zq.keys()))
    X={'fp':D,'int4':enc(D,''),'renorm':enc(D,'renorm')}
    res={k:[] for k in X}; rat=[]
    for qi,q in enumerate(qids):
        rel=set(pos[d] for d in qr[q] if d in pos); S={}
        for k,x in X.items():
            S[k]=np.maximum.reduceat(Q[qi]@x.T,st,axis=1).sum(0); res[k].append(ndcg(np.argsort(-S[k])[:10],rel))
        rat.append((S['int4']-S['fp']).std()/S['fp'].std())
    r={k:np.array(v) for k,v in res.items()}
    print(' ndcg10',{k:round(v.mean(),4) for k,v in r.items()},'ret int4',round(r['int4'].mean()/r['fp'].mean(),3),'renorm',round(r['renorm'].mean()/r['fp'].mean(),3))
    n=len(qids); bs=[]
    for _ in range(1000):
        i=rng.integers(0,n,n); bs.append(r['int4'][i].mean()/r['fp'][i].mean())
    print(' int4 ret boot CI',np.percentile(bs,[2.5,97.5]).round(3),'mean per-query ratio',round(np.mean(rat),2),'median',round(np.median(rat),2))
