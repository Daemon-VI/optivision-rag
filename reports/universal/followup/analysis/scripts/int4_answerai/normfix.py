import json,numpy as np
from scipy.stats import spearmanr
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
cfg={'nfcorpus':R+'f1_text_nfcorpus_a/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_nfcorpus_qrels.json','scifact':R+'f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_scifact_qrels.json'}
def ndcg(top,rel,k=10):
    g=sum(1/np.log2(i+2) for i,d in enumerate(top[:k]) if d in rel); return g/sum(1/np.log2(i+2) for i in range(min(k,len(rel))))
def int4fix(v,fix):
    amax=np.maximum(np.abs(v).max(1),1e-4).astype(np.float16).astype(np.float32)
    q=np.clip(np.round(v/amax[:,None]*7),-7,7)
    if fix:  # norm-preserving scale, stored in the same float16 slot
        dn=np.linalg.norm(q,axis=1)/7*amax; s=(amax*np.linalg.norm(v,axis=1)/np.maximum(dn,1e-9)).astype(np.float16).astype(np.float32)
    else: s=amax
    return (q*(s[:,None]/7)).astype(np.float32)
for ds,qrf in cfg.items():
    z=np.load(f'vec/{ds}_docs.npz'); D=z['D']; lens=z['lens']; ids=[str(i) for i in z['ids']]; pos={d:i for i,d in enumerate(ids)}
    zq=np.load(f'vec/{ds}_queries.npz'); Q=zq['Q']; qids=[str(q) for q in zq['qids']]; qrels=json.load(open(qrf)); starts=np.concatenate([[0],np.cumsum(lens)[:-1]])
    X={'fp':D}; X['int4']=int4fix(D,False); X['int4_normfix']=int4fix(D,True)
    nd=np.linalg.norm(X['int4'],axis=1,keepdims=True); X['fp_x_int4norm(norm error only)']=D*nd
    X['fp_x_shuffled_int4norm']=D*np.random.default_rng(0).permutation(nd)
    res={k:[] for k in X}
    for qi,q in enumerate(qids):
        rel=set(pos[d] for d in qrels[q] if d in pos)
        for k,Xk in X.items():
            M=np.maximum.reduceat(Q[qi]@Xk.T,starts,axis=1).sum(0); res[k].append(ndcg(list(np.argsort(-M)[:10]),rel))
    b=np.mean(res['fp']); print(ds,'fp ndcg10',round(b,4))
    for k in X: print('  ',k,'ndcg10',round(np.mean(res[k]),4),'retention',round(np.mean(res[k])/b,3))
    print('   decoded norm sd int4',float(np.linalg.norm(X['int4'],axis=1).std()),'normfix',float(np.linalg.norm(X['int4_normfix'],axis=1).std()),'bytes unchanged: yes (same float16 scale slot)')
# ratio-vs-retention across the 7 variants x 2 datasets from ablate jsons
pts=[]
for ds in cfg:
    a=json.load(open(f'ablate_{ds}.json'))
    for k,v in a.items():
        if 'ratio' in v: pts.append((ds,k,v['ratio'],v['ret10']))
print('points',len(pts),'spearman(ratio,ret10)',spearmanr([p[2] for p in pts],[p[3] for p in pts])[0])
for p in sorted(pts,key=lambda p:p[2]): print(p[0],p[1],round(p[2],3),round(p[3],3))
