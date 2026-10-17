exec(open('a.py').read().split("rng=np.random")[0])
rng=np.random.default_rng(1)
lab={k:collections.defaultdict(dict) for k in S}; comp=collections.defaultdict(lambda:collections.defaultdict(list))
for r in rows: lab[r[1]][r[2]][r[0]]=r[4]; comp[r[2]][r[1]].append(r[3])
med=lambda l,k:np.median(comp[l][k])
common=[l for l in lab['sci'] if l in lab['doc'] and l in lab['info']]
fixed=[l for l in common if max(med(l,k) for k in S)/min(med(l,k) for k in S)<1.1 and med(l,'doc')>1.5]
print(len(fixed),'fixed-ratio shared pipelines')
def bootdiff(a,b,n=5000):  # a: sci model->val, b: other model->val ; independent bootstrap over models
    A=np.array(list(a.values()));B=np.array(list(b.values()))
    x=[rng.choice(A,len(A)).mean()-rng.choice(B,len(B)).mean() for _ in range(n)]
    return np.percentile(x,[2.5,97.5])
for other in ['doc','info']:
    ds=[]
    for l in fixed:
        s=np.mean(list(lab['sci'][l].values()));o=np.mean(list(lab[other][l].values()));ds.append(s-o)
    ds=np.array(ds);print(other,'pipeline-level mean diff',ds.mean().round(3),'sci lower in',(ds<0).sum(),'/',len(ds),'sci higher:',[l for l,x in zip(fixed,ds) if x>0.0])
    # per-model equal weight across fixed pipelines
    def mm(k):
        ms={m for l in fixed for m in lab[k][l]}
        return {m:np.mean([lab[k][l][m] for l in fixed if m in lab[k][l]]) for m in ms}
    a,b=mm('sci'),mm(other)
    print(' model-mean sci',np.mean(list(a.values())).round(3),other,np.mean(list(b.values())).round(3),'diff CI',bootdiff(a,b).round(3))
    # exclude siblings crudely: drop 'colbertv2.0' duplicate and mxbai edge 17m, LFM2.5, mLateOn
    drop={'colbert-ir/colbertv2.0','mixedbread-ai/mxbai-edge-colbert-v0-17m','LiquidAI/LFM2.5-ColBERT-350M','lightonai/mLateOn'}
    a2={m:v for m,v in a.items() if m not in drop}
    print(' dedup sci',np.mean(list(a2.values())).round(3),'CI',bootdiff(a2,b).round(3), 'n',len(a2))
    # drop worst sci models
    print(' sci per model', sorted([round(v,2) for v in a.values()]))
    print(' other pctiles', np.percentile(list(b.values()),[5,25,50,75,95]).round(2))
    # Mann-Whitney exact-ish by permutation
    A=np.array(list(a.values()));B=np.array(list(b.values()));obs=B.mean()-A.mean();allv=np.r_[A,B]
    p=np.mean([(lambda p:(allv[p[len(A):]].mean()-allv[p[:len(A)]].mean()))(rng.permutation(len(allv)))>=obs for _ in range(20000)]);print(' perm p',p)
# per-pipeline same families
for l in fixed:
    if any(t in l for t in ['int4','binary','lloyd2','project(64)']) and '>' not in l or l in ('project(64) > binary','project(64) > int4(mean)'):
        print(l.ljust(34),*[f"{k}:{np.mean(list(lab[k][l].values())):.3f}/{np.min(list(lab[k][l].values())):.2f}" for k in S])
# recall@1 claim? skip; dtype confound: float16 sci models
