import sys,glob,json,string,numpy as np,torch,pandas as pd
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open
ds=sys.argv[1]
torch.set_num_threads(8)
snap=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/models--answerdotai--answerai-colbert-small-v1/snapshots/*')[0]
tok=AutoTokenizer.from_pretrained(snap); m=AutoModel.from_pretrained(snap).eval()
W=safe_open(snap+'/model.safetensors','pt').get_tensor('linear.weight')
def base(name):
    return glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/datasets--BeIR--%s/snapshots/*'%name)[0]
corp=pd.read_parquet(glob.glob(base(ds)+'/corpus/*.parquet')[0])
qs=pd.read_parquet(glob.glob(base(ds)+'/queries/*.parquet')[0])
qrels=json.load(open(sys.argv[2]))
ids=[str(i) for i in corp['_id']]; texts=[(t+' '+x).strip() for t,x in zip(corp['title'],corp['text'])]
punct=set(tok.convert_tokens_to_ids([c for c in string.punctuation if c in tok.vocab]))
@torch.no_grad()
def enc(batch_ids,att):
    h=m(input_ids=batch_ids,attention_mask=att).last_hidden_state
    o=h@W.T
    return torch.nn.functional.normalize(o,dim=-1)
# docs
docs=[];order=np.argsort([len(t) for t in texts])
out=[None]*len(texts)
for s in range(0,len(texts),32):
    idx=order[s:s+32]
    e=tok([texts[i] for i in idx],truncation=True,max_length=299,padding=True,return_tensors='pt')
    ii=e['input_ids'].clone(); ii[:,1:]=torch.cat([torch.full((ii.shape[0],1),1),e['input_ids'][:,1:-0 or None][:, :-1]],1) if False else ii[:,1:]
    # insert marker: shift tokens right by one after CLS
    ids_=torch.cat([e['input_ids'][:,:1],torch.full((len(idx),1),1),e['input_ids'][:,1:]],1)
    att=torch.cat([e['attention_mask'][:,:1],torch.ones(len(idx),1,dtype=torch.long),e['attention_mask'][:,1:]],1)
    o=enc(ids_,att)
    for r,i in enumerate(idx):
        keep=att[r].bool().clone()
        for p in range(ids_.shape[1]):
            if int(ids_[r,p]) in punct: keep[p]=False
        out[i]=o[r][keep].numpy().astype(np.float32)
    if s%640==0: print(s,flush=True)
lens=np.array([len(o) for o in out]); print('vpd',lens.mean())
D=np.concatenate(out); np.savez(f'vec/{ds}_docs.npz',D=D,lens=lens,ids=np.array(ids))
qids=[q for q in qrels]; qt={str(a):b for a,b in zip(qs['_id'],qs['text'])}
Q=[]
for q in qids:
    e=tok(qt[q],truncation=True,max_length=31,return_tensors='pt')
    x=e['input_ids'][0].tolist(); x=[x[0],0]+x[1:]   # CLS,[Q],...,SEP
    x=x+[tok.mask_token_id]*(32-len(x)); x=x[:32]
    t=torch.tensor([x]); o=enc(t,torch.ones_like(t))
    Q.append(o[0].numpy())
np.savez(f'vec/{ds}_queries.npz',Q=np.stack(Q),qids=np.array(qids))
