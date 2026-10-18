import sys,json,numpy as np
ds=sys.argv[1]; qrf=sys.argv[2]
z=np.load(f'vec/{ds}_docs.npz'); D=z['D'].astype(np.float32); lens=z['lens']; ids=[str(i) for i in z['ids']]
zq=np.load(f'vec/{ds}_queries.npz'); Q=zq['Q'].astype(np.float32); qids=[str(q) for q in zq['qids']]
qrels=json.load(open(qrf)); pos={d:i for i,d in enumerate(ids)}
starts=np.concatenate([[0],np.cumsum(lens)[:-1]])
def int4(v,center=None,bits=4):
    L=2**(bits-1)-1
    if center is not None: v=v-center
    amax=np.maximum(np.abs(v).max(1),1e-4).astype(np.float16).astype(np.float32)
    q=np.clip(np.round(v/amax[:,None]*L),-L,L)
    out=q*(amax[:,None]/L)
    if center is not None: out=out+center
    return out.astype(np.float32)
mu=D.mean(0); muhat=mu/np.linalg.norm(mu)
V={'fp':D}
V['int4']=int4(D); V['int4mean']=int4(D,mu); V['int8']=int4(D,None,8)
e=V['int4']-D
epar=(e@muhat)[:,None]*muhat[None]
V['fp+e_perp']=D+(e-epar); V['fp+e_par']=D+epar
rng=np.random.default_rng(0)
# same-norm random-direction error
r=rng.standard_normal(e.shape).astype(np.float32); r/=np.linalg.norm(r,axis=1,keepdims=True); V['fp+e_randdir']=D+r*np.linalg.norm(e,axis=1,keepdims=True)
n=np.linalg.norm(V['int4'],axis=1,keepdims=True); V['int4_renorm']=V['int4']/n
print('decoded norm mean/sd',n.mean(),n.std(),' sd of e.muhat',(e@muhat).std(),' mean of e.muhat',(e@muhat).mean(),' |e| mean',np.linalg.norm(e,axis=1).mean())
def ndcg(rank_ids,rel,k):
    g=sum(1/np.log2(i+2) for i,d in enumerate(rank_ids[:k]) if d in rel)
    idg=sum(1/np.log2(i+2) for i in range(min(k,len(rel)))); return g/idg
res={k:{'n5':[],'n10':[],'r1':[]} for k in V}; stats={k:[] for k in V}
fpS={}
for qi,q in enumerate(qids):
    rel=set(pos[d] for d in qrels[q] if d in pos)
    for k,X in V.items():
        S=Q[qi]@X.T
        M=np.maximum.reduceat(S,starts,axis=1).sum(0)
        if k=='fp': M0=M
        else: stats[k].append((np.std(M-M0),np.std(M0),np.corrcoef(M-M0,np.log(lens))[0,1], np.mean(M-M0)))
        top=np.argsort(-M)[:10]
        res[k]['n5'].append(ndcg(list(top),rel,5)); res[k]['n10'].append(ndcg(list(top),rel,10)); res[k]['r1'].append(float(top[0] in rel))
    if qi%50==0: print(qi,flush=True)
b=np.mean(res['fp']['n10'])
print('baseline ndcg10',b,'ndcg5',np.mean(res['fp']['n5']))
out={}
for k in V:
    m10=np.mean(res[k]['n10']); s=np.array(stats[k]) if stats[k] else None
    line=dict(ndcg10=m10,ret10=m10/b,ret5=np.mean(res[k]['n5'])/np.mean(res['fp']['n5']),ret_r1=np.mean(res[k]['r1'])/np.mean(res['fp']['r1']))
    if s is not None: line.update(sd_err=s[:,0].mean(),sd_true=s[:,1].mean(),ratio=(s[:,0]/s[:,1]).mean(),corr_err_loglen=s[:,2].mean(),mean_err=s[:,3].mean())
    out[k]=line; print(k,{a:round(float(v),4) for a,v in line.items()})
json.dump(out,open(f'ablate_{ds}.json','w'),indent=1,default=float)
