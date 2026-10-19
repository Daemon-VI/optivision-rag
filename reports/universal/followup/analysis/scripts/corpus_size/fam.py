import json,re,numpy as np
exec(open('followup/analysis/scripts/corpus_size/pairs.py').read().split("M='ndcg@5'")[0])
M='ndcg@5'
def fam(l,dim):
    m=re.findall(r'project\((\d+)',l)
    pd=int(m[0]) if m else dim
    if 'random' in l: return 'randproj'
    if m and pd/dim<=0.25 and not (l.count('project(128)')): return 'proj<=25%dim'
    if m and pd/dim<=0.5 and 'project(128)' not in l: return 'proj50%'
    if 'merge' in l and 'binary' in l: return 'merge+binary'
    if 'merge' in l: return 'merge+codec'
    return 'codec/other'
allrows=[]
for n,a,b in P:
    A=json.load(open(R+a));B=json.load(open(R+b));dim=A['baseline']['dim']
    ra={r['label']:r for r in A['rows']}
    for r in B['rows']:
        o=ra.get(r['label'])
        if o: allrows.append((n,fam(r['label'],dim),r['label'],o['retention:'+M],r['retention:'+M],r['compression_vs_float32'],o['retention_ci:'+M]))
import collections
for n in ['colbertv2','answerai','colSmol','colpali',None]:
    print('\n',n)
    g=collections.defaultdict(list)
    for x in allrows:
        if n is None or x[0]==n: g[x[1]].append(x[4]-x[3])
    for k,v in sorted(g.items()): print('  %-14s n=%2d mean d=%+.3f min=%+.3f'%(k,len(v),np.mean(v),min(v)))
# among rows with original retention>=0.9 (usable): d
print('\nusable rows (orig ret>=0.9)')
for n in ['colbertv2','answerai','colSmol','colpali']:
    v=[(x[4]-x[3],x[2],x[5]) for x in allrows if x[0]==n and x[3]>=0.9]
    d=np.array([a for a,_,_ in v]);print(n,len(v),'mean %+.3f min %+.3f'%(d.mean(),d.min()), 'n<-0.03',(d<-0.03).sum(), [ (round(a,3),l[:40],round(c)) for a,l,c in sorted(v)[:3]])
# correlation of d with orig loss and with compression
for n in ['colbertv2','answerai','colSmol','colpali']:
    X=[x for x in allrows if x[0]==n]
    d=np.array([x[4]-x[3] for x in X]);c=np.log([x[5] for x in X]);lo=np.array([1-x[3] for x in X])
    from scipy.stats import spearmanr
    print(n,'spearman d~logcomp %.2f  d~origloss %.2f'%(spearmanr(d,c)[0],spearmanr(d,-lo)[0]))
# relative loss: new loss / orig loss for lossy rows
for n in ['colbertv2','answerai','colSmol','colpali']:
    X=[x for x in allrows if x[0]==n and x[3]<0.9 and x[3]>0.3]
    print(n,'lossy rows(0.3-0.9) n',len(X),'median new/orig retention %.2f'%np.median([x[4]/x[3] for x in X]))
