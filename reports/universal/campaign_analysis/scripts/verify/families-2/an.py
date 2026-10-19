import json,re,sys,numpy as np
rng=np.random.default_rng(1)
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
DS=['vidore/docvqa_test_subsampled','vidore/infovqa_test_subsampled']
img={k:v for k,v in m.items() if all(d in v for d in DS)}
def fam(k):
    s=k.lower()
    if 'colpali' in s: return 'colpali'
    if 'neomme' in s: return 'neomme'
    if 'colsmol' in s: return 'smolvlm'
    if 'tomoro' in s or 'sauerkraut' in s: return 'qwen3vl'
    if any(x in s for x in ['evie','vultron','colqwen3.5']): return 'qwen3.5'
    if 'colqwen2.5' in s or 'colnomic' in s or 'evoqwen' in s: return 'colqwen2.5'
    if 'colqwen2-' in s or 'colqwen-omni' in s: return 'colqwen2'
    return None
twin=lambda k: k.endswith('-hf') or k.endswith('-merged')
fams={k:fam(k) for k in img}
print({k:f for k,f in fams.items() if f is None})
def rows(k,d): return {r['label']:r['retention'] for r in img[k][d]['rows']}
import collections
cc=collections.Counter()
for k in img:
    if fams[k] is None: continue
    for d in DS:
        for l in rows(k,d): cc[l]+=1
common=sorted(l for l,n in cc.items() if n>=62 and l not in ('float32','float16','int4'))
def build(labels,drop_twins=True):
    ks=[k for k in img if fams[k] and all(l in rows(k,d) for d in DS for l in labels) and not(drop_twins and twin(k))]
    X=np.array([[[rows(k,d)[l] for l in labels] for d in DS] for k in ks])  # M,2,L
    f=np.array([fams[k] for k in ks])
    # keep families >=2
    u,c=np.unique(f,return_counts=True); keep=np.isin(f,u[c>=2])
    return [k for k,t in zip(ks,keep) if t],X[keep],f[keep]
def decomp(X,f):
    M,D,L=X.shape
    Y=X-X.mean((0,1),keepdims=True)           # per label centre
    dsm=Y.mean(0,keepdims=True); ssD=M*((dsm**2).sum())
    R=Y-dsm; Mm=R.mean(1)                     # M,L  model means
    fm=np.zeros_like(Mm)
    for g in np.unique(f): fm[f==g]=Mm[f==g].mean(0)
    ssF=D*((fm**2).sum()); ssM=D*(((Mm-fm)**2).sum())
    ssI=((R-Mm[:,None,:])**2).sum()
    tot=ssD+ssF+ssM+ssI
    return np.array([ssD,ssF,ssM,ssI])/tot
def run(labels,drop_twins=True,tag=''):
    ks,X,f=build(labels,drop_twins)
    print('==',tag,'models',len(ks),'labels',len(labels),dict(zip(*np.unique(f,return_counts=True))))
    s=decomp(X,f); print(' shares D/F/M/I',s.round(3))
    perm=[]
    for _ in range(2000): perm.append(decomp(X,rng.permutation(f))[1])
    perm=np.array(perm); print(' fam share perm null mean %.3f p=%.4f'%(perm.mean(),(1+(perm>=s[1]).sum())/2001))
    # LOO
    M=len(ks); eg=[];ef=[];et=[]
    for i in range(M):
        oth=[j for j in range(M) if j!=i and not (f[j]==f[i] and (ks[j].replace('-hf','').replace('-merged','')==ks[i].replace('-hf','').replace('-merged','')))]
        sib=[j for j in oth if f[j]==f[i]]
        if not sib: eg.append(np.nan);ef.append(np.nan);et.append(np.nan);continue
        for d in range(2):
            pass
        g=X[oth].mean(0); fa=X[sib].mean(0)
        # transplant: global mean on d + own deviation from family on other d
        dev=X[i]-fa
        tp=np.stack([g[0]+dev[1],g[1]+dev[0]])  # deviation vs family mean other dataset
        eg.append(((X[i]-g)**2).sum()); ef.append(((X[i]-fa)**2).sum()); et.append(((X[i]-tp)**2).sum())
    eg,ef,et=map(np.array,(eg,ef,et)); ok=~np.isnan(eg)
    r=ef[ok].sum()/eg[ok].sum()
    bs=[]
    idx=np.where(ok)[0]
    for _ in range(4000):
        b=rng.choice(idx,len(idx)); bs.append(ef[b].sum()/eg[b].sum())
    print(' LOO fam/global SE ratio %.3f CI %s'%(r,np.percentile(bs,[2.5,97.5]).round(2)))
    # drop extreme model
    infl=[(ef[ok].sum()-ef[j])/(eg[ok].sum()-eg[j]) for j in idx]
    print(' jackknife range %.2f-%.2f'%(min(infl),max(infl)))
    # per-family RMSE
    for g in np.unique(f):
        s_=[i for i in idx if f[i]==g]
        print('  ',g,len(s_),'fam rmse %.4f glob rmse %.4f'%(np.sqrt(ef[s_].sum()/(2*len(s_)*len(labels))),np.sqrt(eg[s_].sum()/(2*len(s_)*len(labels)))))
    # deviation persistence
    Y=X.copy()
    for g in np.unique(f):
        Y[f==g]-=X[f==g].mean(0)
    Y=Y-Y.mean(0)*0
    dd=Y[:,0,:].ravel(); di=Y[:,1,:].ravel()
    # per-model mean deviation over labels
    c=np.corrcoef(Y[:,0,:].ravel(),Y[:,1,:].ravel())[0,1]
    def stat(f_):
        Z=X.copy()
        for g in np.unique(f_): Z[f_==g]-=X[f_==g].mean(0)
        return np.corrcoef(Z[:,0,:].ravel(),Z[:,1,:].ravel())[0,1]
    # permute: shuffle infovqa rows among models (breaks model identity)
    pc=[]
    for _ in range(2000):
        p=rng.permutation(len(ks)); Z=Y.copy(); a=Z[:,0,:].ravel(); b=Z[p,1,:].ravel(); pc.append(np.corrcoef(a,b)[0,1])
    pc=np.array(pc); print(' dev corr %.3f perm p(two-sided)=%.3f  (shuffled within-family-demeaned across models)'%(c,(1+(np.abs(pc)>=abs(c)).sum())/2001))
    # variant: shuffle only within family
    pc=[]
    for _ in range(2000):
        p=np.arange(len(ks))
        for g in np.unique(f):
            ii=np.where(f==g)[0]; p[ii]=rng.permutation(ii)
        pc.append(np.corrcoef(Y[:,0,:].ravel(),Y[p,1,:].ravel())[0,1])
    pc=np.array(pc); print(' within-family-shuffle p=%.3f'%((1+(pc>=c).sum())/2001))
    # 1-NN
    P=X.reshape(M,-1)
    hit=0
    for i in range(M):
        dist=((P-P[i])**2).sum(1); dist[i]=np.inf
        for j in range(M):
            if ks[j].replace('-hf','').replace('-merged','')==ks[i].replace('-hf','').replace('-merged','') and j!=i: dist[j]=np.inf
        hit+= f[np.argmin(dist)]==f[i]
    ch=np.mean([ ((f==f[i]).sum()-1)/(M-1) for i in range(M)])
    print(' 1NN same-family %.2f chance~%.2f'%(hit/M,ch))
lab_all=common
run(lab_all,True,'all common labels, twins dropped')
run(lab_all,False,'all common labels, twins kept')
import random
lab17=[l for l in common if ('0.25' not in l)][:17]
print(len(lab17))
run(lab17,True,'subset')

print('--- extra: bootstrap CI of deviation corr, per-model mean deviation, transplant ratio')
for tag,labs in [('29',common),('17',lab17)]:
    ks,X,f=build(labs,True); M=len(ks)
    def dev(X,f):
        Y=X.copy()
        for g in np.unique(f): Y[f==g]-=X[f==g].mean(0)
        return Y
    Y=dev(X,f); c=np.corrcoef(Y[:,0].ravel(),Y[:,1].ravel())[0,1]
    bs=[]
    for _ in range(2000):
        b=rng.integers(0,M,M); fb=f[b]
        # need families with >=2 distinct draws
        Yb=dev(X[b],fb)
        if np.isnan(Yb).any(): continue
        bs.append(np.corrcoef(Yb[:,0].ravel(),Yb[:,1].ravel())[0,1])
    print(tag,'corr %.3f boot CI'%c,np.nanpercentile(bs,[2.5,97.5]).round(2))
    # per-model mean deviation (over labels) correlation
    a=Y[:,0].mean(1);b_=Y[:,1].mean(1); print(' mean-dev corr',np.corrcoef(a,b_)[0,1].round(3))
    # transplant: dataset-wise global-mean + own other-dataset deviation (vs global mean)
    ratio_num=0;ratio_den=0
    G=X.mean(0)
    for i in range(M):
        o=[j for j in range(M) if j!=i]; g=X[o].mean(0)
        dv=X[i]-g
        tp=np.stack([g[0]+dv[1],g[1]+dv[0]])
        ratio_num+=((X[i]-tp)**2).sum(); ratio_den+=((X[i]-g)**2).sum()
    print(' transplant(vs global) SE ratio %.2f (2.0 if no persistence)'%(ratio_num/ratio_den))
