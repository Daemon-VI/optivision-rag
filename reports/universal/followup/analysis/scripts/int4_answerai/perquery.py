import sys,glob,json,numpy as np,pandas as pd
from transformers import AutoTokenizer
from scipy.stats import spearmanr
snap=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/models--answerdotai--answerai-colbert-small-v1/snapshots/*')[0]
tok=AutoTokenizer.from_pretrained(snap)
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
cfg={'nfcorpus':R+'f1_text_nfcorpus_a/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_nfcorpus_qrels.json','scifact':R+'f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_scifact_qrels.json'}
def int4(v,center=None):
    if center is not None: v=v-center
    amax=np.maximum(np.abs(v).max(1),1e-4).astype(np.float16).astype(np.float32)
    out=np.clip(np.round(v/amax[:,None]*7),-7,7)*(amax[:,None]/7)
    if center is not None: out=out+center
    return out.astype(np.float32)
def ndcg(top,rel,k=10):
    g=sum(1/np.log2(i+2) for i,d in enumerate(top[:k]) if d in rel); return g/sum(1/np.log2(i+2) for i in range(min(k,len(rel))))
allrows={}
for ds,qrf in cfg.items():
    z=np.load(f'vec/{ds}_docs.npz'); D=z['D']; lens=z['lens']; ids=[str(i) for i in z['ids']]; pos={d:i for i,d in enumerate(ids)}
    zq=np.load(f'vec/{ds}_queries.npz'); Q=zq['Q']; qids=[str(q) for q in zq['qids']]; qrels=json.load(open(qrf))
    base=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/datasets--BeIR--%s/snapshots/*'%ds)[0]
    qs=pd.read_parquet(glob.glob(base+'/queries/*.parquet')[0]); qt={str(a):b for a,b in zip(qs['_id'],qs['text'])}
    starts=np.concatenate([[0],np.cumsum(lens)[:-1]])
    X={'fp':D,'int4':int4(D)}; X['renorm']=X['int4']/np.linalg.norm(X['int4'],axis=1,keepdims=True)
    rows=[]
    for qi,q in enumerate(qids):
        n=min(len(tok(qt[q],truncation=True,max_length=31)['input_ids'])+1,32)
        qv=Q[qi]; G=qv@qv.T; coh=(G.sum()-32)/(32*31)
        rel=set(pos[d] for d in qrels[q] if d in pos); r=dict(n_real=n,qcoh=coh,nrel=len(rel))
        for k,Xk in X.items():
            M=np.maximum.reduceat(qv@Xk.T,starts,axis=1).sum(0)
            if k=='fp': M0=M; r['sd_true']=M0.std(); top1=np.sort(M0)[::-1]; r['gap10']=top1[0]-top1[9]
            else: r['sd_err_'+k]=(M-M0).std()
            r['nd_'+k]=ndcg(list(np.argsort(-M)[:10]),rel)
        rows.append(r)
    df=pd.DataFrame(rows); df['ratio']=df.sd_err_int4/df.sd_true; df['drop']=df.nd_fp-df.nd_int4; allrows[ds]=df
    df.to_csv(f'perq_{ds}.csv',index=False)
    print('==',ds,'n queries',len(df))
    print('mean n_real tokens',df.n_real.mean().round(1),'median',df.n_real.median(),' mean query pairwise cos',df.qcoh.mean().round(3),' sd_true',df.sd_true.mean().round(4),' sd_err int4',df.sd_err_int4.mean().round(4),' renorm',df.sd_err_renorm.mean().round(4))
    print('ndcg10 fp/int4/renorm',df.nd_fp.mean().round(3),df.nd_int4.mean().round(3),df.nd_renorm.mean().round(3))
    for c in ['n_real','qcoh','nrel','sd_true','ratio']:
        print('  spearman(drop, %s)'%c, round(spearmanr(df['drop'],df[c])[0],3))
    df['q']=pd.qcut(df.n_real,4,duplicates='drop')
    print(df.groupby('q',observed=True).agg(n=('nd_fp','size'),nreal=('n_real','mean'),qcoh=('qcoh','mean'),sd_true=('sd_true','mean'),ratio=('ratio','mean'),fp=('nd_fp','mean'),i4=('nd_int4','mean'),rn=('nd_renorm','mean')).round(3))
