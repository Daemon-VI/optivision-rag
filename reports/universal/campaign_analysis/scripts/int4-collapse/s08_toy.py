"""SYNTHETIC toy (not data about real models): re-implements the package's per-vector int4 / int8 / binary rules in numpy and asks
how retention of a MaxSim toy retriever depends on (a) dimension d, (b) strength a of a SHARED offset direction added to every token,
(c) whether the offset is concentrated on one coordinate ('sparse') or spread over all coordinates ('dense'). Quantiser code mirrors
src/optivision/stages/quantize.py (levels -7..7 times per-vector max|x|/7, float16 scale)."""
import numpy as np,json,sys
rng=np.random.default_rng(1)
def unit(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
def q_int4(v,mean=None):
    v=v.astype(np.float32); mu=0 if mean is None else mean
    r=v-mu; amax=np.maximum(np.abs(r).max(1),np.finfo(np.float16).tiny).astype(np.float16).astype(np.float32)
    q=np.clip(np.round(r/amax[:,None]*7),-7,7); return q*amax[:,None]/7+mu
def q_int8(v):
    amax=np.maximum(np.abs(v).max(1),np.finfo(np.float16).tiny).astype(np.float16).astype(np.float32)
    q=np.clip(np.round(v/amax[:,None]*127),-127,127); return q*amax[:,None]/127
def q_bin(v): return np.where(v>=0,1.0,-1.0).astype(np.float32)
def make(d,a,kind,nd=150,nt=40,nq=150,seed=0):
    r=np.random.default_rng(seed)
    topics=unit(r.standard_normal((nd,3,d)))
    if kind=='sparse': m=np.zeros(d); m[0]=1
    else: m=unit(r.standard_normal(d))
    docs=[];
    for i in range(nd):
        t=r.integers(0,3,nt); b=unit(topics[i,t]+0.7*unit(r.standard_normal((nt,d)))); docs.append(unit(b+a*m))
    D=np.stack(docs).astype(np.float32)
    qi=r.integers(0,nd,nq); Q=[]
    for i in qi:
        t=r.integers(0,3,5); b=unit(topics[i,t]+0.9*unit(r.standard_normal((5,d)))); Q.append(unit(b+a*m))
    return D,np.stack(Q).astype(np.float32),qi
def score(D,Q):
    nd,nt,d=D.shape; S=np.einsum('qkd,ntd->qknt',Q,D,optimize=True) if False else None
    flat=D.reshape(-1,d); s=Q.reshape(-1,d)@flat.T; s=s.reshape(Q.shape[0],Q.shape[1],nd,nt).max(3).sum(1); return s
def acc(D,Q,qi): return float((score(D,Q).argmax(1)==qi).mean())
out=[]
for d in [48,128,640,2560]:
    for kind in ['sparse','dense']:
        for a in [0,0.5,1,2,4,8]:
            D,Q,qi=make(d,a,kind); nd,nt,_=D.shape
            base=acc(D,Q,qi)
            mu=D.reshape(-1,d).mean(0)
            def tr(f): return acc(f(D.reshape(-1,d)).reshape(D.shape),f(Q.reshape(-1,d)).reshape(Q.shape),qi)
            r4=tr(q_int4)/base; r4m=tr(lambda x:q_int4(x,mu))/base; r8=tr(q_int8)/base; rb=tr(q_bin)/base
            zero=float((np.abs(q_int4(D.reshape(-1,d)))==0).mean())
            out.append(dict(d=d,kind=kind,a=a,share=float(a/np.sqrt(1+a*a)),base=base,int4=r4,int4m=r4m,int8=r8,binary=rb,zero_frac_int4=zero))
            print(out[-1],flush=True)
json.dump(out,open('toy.json','w'))
