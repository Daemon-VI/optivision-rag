from load import *
from scipy import stats
rng=np.random.default_rng(1)
def proj(d):
    out={}
    for r in d['rows']:
        st=r['pipeline']['stages']
        if len(st)==1 and st[0]['stage']=='project':
            out.setdefault((st[0].get('method','pca'),st[0]['dim']),[]).append(r)
    return out
pairs=[];dup=0
for (m,ds),v in sorted(U.items()):
    d=v[0][1]; D=d['baseline']['dim']; P=proj(d)
    for (me,k),rs in P.items():
        if len(rs)>1: dup+=1
        if me=='random' and ('pca',k) in P and D>128:
            a=P[('pca',k)][0]; b=rs[0]
            pairs.append((m,ds,D,k,a[M],b[M],a['retention_ci:ndcg@5'],b['retention_ci:ndcg@5'],d['baseline'].get('dtype')))
print('dup',dup,'pairs',len(pairs))
diff=np.array([(p[4]-p[5])*100 for p in pairs])
by={}
for p in pairs: by.setdefault(p[0],[]).append((p[4]-p[5])*100)
keys=list(by); mm=np.array([np.mean(by[k]) for k in keys])
print(len(keys),'models; units',len(diff),'mean',diff.mean(),'ahead',(diff>0).sum())
print('model means',{k[:30]:round(v,2) for k,v in zip(keys,mm)})
cb=[np.concatenate([by[keys[i]] for i in rng.integers(0,len(keys),len(keys))]).mean() for _ in range(20000)]
print('cluster CI',np.percentile(cb,[2.5,97.5]))
print('models ahead',(mm>0).sum(),'sign p',stats.binomtest(int((mm>0).sum()),len(mm)).pvalue,'units sign p',stats.binomtest(int((diff>0).sum()),len(diff)).pvalue)
print('wilcoxon models',stats.wilcoxon(mm).pvalue,'units',stats.wilcoxon(diff).pvalue, 'ties',(diff==0).sum())
for D in sorted({p[2] for p in pairs}):
    x=[(p[4]-p[5])*100 for p in pairs if p[2]==D]; print('D',D,len(x),round(np.mean(x),2))
for ds in['docvqa','infovqa']:
    x=[(p[4]-p[5])*100 for p in pairs if p[1]==ds]; print(ds,len(x),round(np.mean(x),2))
for k in keys:
    print('LOO',k[:35],round(np.mean([x for j in keys if j!=k for x in by[j]]),2))
fam=lambda m: m.split('--')[-1].lower().split('-')[0].replace('colqwen3.5','tomoro-like')
for p in pairs:
    if p[3] in(640,): print(p[:6])
big=[k for k in keys if 'emotron' in k or 'EVIE' in k]
print('<=640d-only models',np.mean([x for k in keys if k not in big for x in by[k]]))
# 2048+ group stats
x=[(p[4]-p[5])*100 for p in pairs if p[2]>=2048]; print('2048+',len(x),np.mean(x),'mean w/o nemo4b docvqa',np.mean([(p[4]-p[5])*100 for p in pairs if p[2]>=2048 and not(p[0].startswith('nemotron-colembed-vl-4b') and p[1]=='docvqa')]))
# CI per pair overlap
nonov=sum(1 for p in pairs if p[6][0]>p[7][1]); rev=sum(1 for p in pairs if p[7][0]>p[6][1]); print('CI-separated PCA>rand',nonov,'rand>PCA',rev)
import collections
print('baseline dtype', collections.Counter(str(p[8]) for p in pairs))
# family-level
famd={}
for k in keys:
    f='nemotron' if 'emotron' in k else 'evie' if 'EVIE' in k else 'tomoro' if 'omoro' in k else 'vultron' if 'ultron' in k else 'webai' if 'webAI' in k else k
    famd.setdefault(f,[]).extend(by[k])
fmm=np.array([np.mean(v) for v in famd.values()]); print({k:(len(v),round(np.mean(v),2)) for k,v in famd.items()}); print('fam mean',fmm.mean(),(fmm>0).sum(),len(fmm),stats.binomtest(int((fmm>0).sum()),len(fmm)).pvalue)
cb=[fmm[rng.integers(0,len(fmm),len(fmm))].mean() for _ in range(20000)]; print('fam boot',np.percentile(cb,[2.5,97.5]))
# retention>1 noise: with clipping at 1
print('clipped at 1 mean',np.mean([(min(p[4],1)-min(p[5],1))*100 for p in pairs]))
