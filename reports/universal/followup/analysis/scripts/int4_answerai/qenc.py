import sys,glob,json,numpy as np,torch,pandas as pd
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open
snap=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/models--answerdotai--answerai-colbert-small-v1/snapshots/*')[0]
tok=AutoTokenizer.from_pretrained(snap); m=AutoModel.from_pretrained(snap).eval()
W=safe_open(snap+'/model.safetensors','pt').get_tensor('linear.weight')
for ds in sys.argv[1:]:
    base=glob.glob(r'C:/Users/Lenovo/.cache/huggingface/hub/datasets--BeIR--%s/snapshots/*'%ds)[0]
    qs=pd.read_parquet(glob.glob(base+'/queries/*.parquet')[0]); qt={str(a):b for a,b in zip(qs['_id'],qs['text'])}
    qids=[str(q) for q in np.load(f'vec/{ds}_queries.npz')['qids']]; Q=[]
    for q in qids:
        x=tok(qt[q],truncation=True,max_length=31)['input_ids']; x=[x[0],0]+x[1:]; n=len(x); x=(x+[tok.mask_token_id]*(32-n))[:32]
        t=torch.tensor([x]); att=torch.ones_like(t); att[0,n:]=0
        with torch.no_grad(): h=m(input_ids=t,attention_mask=att).last_hidden_state
        Q.append(torch.nn.functional.normalize(h@W.T,dim=-1)[0].numpy())
    np.savez(f'vec/{ds}_queries.npz',Q=np.stack(Q),qids=np.array(qids))
