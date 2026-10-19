import sys,glob,json,numpy as np,torch,pandas as pd
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open
torch.set_num_threads(8)
snap=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/models--answerdotai--answerai-colbert-small-v1/snapshots/*')[0]
tok=AutoTokenizer.from_pretrained(snap); m=AutoModel.from_pretrained(snap).eval()
W=safe_open(snap+'/model.safetensors','pt').get_tensor('linear.weight')
base=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/datasets--BeIR--nfcorpus/snapshots/*')[0]
qs=pd.read_parquet(glob.glob(base+'/queries/*.parquet')[0]); qt={str(a):b for a,b in zip(qs['_id'],qs['text'])}
qrels=json.load(open('C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/f1_text_nfcorpus_a/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_nfcorpus_qrels.json'))
z=np.load('vec/nfcorpus_docs.npz'); D=z['D']; lens=z['lens']; ids=[str(i) for i in z['ids']]; pos={d:i for i,d in enumerate(ids)}
starts=np.concatenate([[0],np.cumsum(lens)[:-1]])
def enc(q,maskatt,lower):
    t=qt[q].lower() if lower else qt[q]
    x=tok(t,truncation=True,max_length=31)['input_ids']; x=[x[0],0]+x[1:]; n=len(x)
    x=(x+[tok.mask_token_id]*(32-n))[:32]
    ids_=torch.tensor([x]); att=torch.ones_like(ids_)
    if maskatt: att[0,n:]=0
    with torch.no_grad(): h=m(input_ids=ids_,attention_mask=att).last_hidden_state
    return torch.nn.functional.normalize(h@W.T,dim=-1)[0].numpy()
def ndcg(top,rel,k=10):
    g=sum(1/np.log2(i+2) for i,d in enumerate(top[:k]) if d in rel); return g/sum(1/np.log2(i+2) for i in range(min(k,len(rel))))
for name,ma,lo in [('base',False,False),('mask_not_attended',True,False)]:
    sc=[]
    for q in qrels:
        Q=enc(q,ma,lo); rel=set(pos[d] for d in qrels[q] if d in pos)
        M=np.maximum.reduceat(Q@D.T,starts,axis=1).sum(0); sc.append(ndcg(list(np.argsort(-M)[:10]),rel))
    print(name,np.mean(sc),flush=True)
