import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.stats import norm
rng=np.random.default_rng(1); B=4000
U=load()
units={k:u for k,u in U.items() if 'plus10000' not in k[1]}
TEXT=[k for k in units if k[1] in('scifact','nfcorpus')]; PAGE=[k for k in units if 'subsampled' in k[1]]; SCI=[k for k in units if k[1]=='scidocs']
print(len(units),len(TEXT),len(PAGE),len(SCI))
sim={}  # key-> dict label -> array of simulated @5 retention = obs@5 + (boot@10 ret - obs@10 ret)
for k,u in units.items():
    d=u['d']; pq=d['per_query_ndcg@10']; n=len(pq['float32']); idx=rng.integers(0,n,(B,n))
    f=np.array(pq['float32']); fb=f[idx].mean(1)
    s={}
    for l in ['int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']:
        x=np.array(pq[l]); rb=x[idx].mean(1)/fb; ro=x.mean()/f.mean()
        s[l]=row(d,l)['retention:ndcg@5']+(rb-ro)
        s[l+'_sd']=rb.std()
    sim[k]=s
def obs(k,l): return row(units[k]['d'],l)['retention:ndcg@5']
def lo(k,l): return row(units[k]['d'],l)['retention_ci:ndcg@5'][0]
print('\n== interval widths (ndcg@5 retention CI width, int4norm(mean)) by dataset')
import collections
W=collections.defaultdict(list)
for k in units:
    for l in ['int4norm','int4norm(mean)','int8(per_vector)']:
        ci=row(units[k]['d'],l)['retention_ci:ndcg@5']; W[(k[1][:12],l)].append(ci[1]-ci[0])
for (ds,l),v in sorted(W.items()):
    nq=[units[k]['d']['n_queries'] for k in units if k[1][:12]==ds][0]
    print('%-14s nq=%-5d %-16s units=%2d width median %.3f min %.3f max %.3f'%(ds,nq,l,len(v),np.median(v),min(v),max(v)))
print('\n== page recall@1 / recall@5 (float32), and per-page-unit int8 changed queries')
for k in PAGE:
    b=units[k]['d']['baseline']; pq=units[k]['d']['per_query_ndcg@10']
    ch=(abs(np.array(pq['int8(per_vector)'])-np.array(pq['float32']))>1e-9).sum()
    print('%-34s r@1 %.3f r@5 %.3f  nq %d  int8 changed %d'%(k[0][:22]+'/'+k[1][:8],b['recall@1'],b['recall@5'],units[k]['d']['n_queries'],ch))
print('\n== Q6 flip: P(diff < -0.01) per unit (simulated), observed diff @5; both pairs')
anyfail=np.zeros(B,bool); tabl=[]
for k in units:
    for a,b in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
        dd=sim[k][a]-sim[k][b]; p=(dd<-0.01).mean(); anyfail|=(dd<-0.01)
        tabl.append((p,k,a,obs(k,a)-obs(k,b)))
for p,k,a,o in sorted(tabl,reverse=True)[:8]: print('P(fail)=%.3f  %s %s obs diff %+.4f'%(p,k,a,o))
print('P(at least one of 92 pair-checks fails) = %.3f ; sum of unit probs %.2f'%(anyfail.mean(),sum(t[0] for t in tabl)))
print('Q6 observed: failing pair-checks (diff<-0.01):',[(k,a,round(o,4)) for p,k,a,o in tabl if o<-0.01])
print('\n== Q2 (10 norm-noise text units: answerai,mLateOn,mxbai-edge 17m/32m,GTE x scifact,nfcorpus), int4norm point >=0.95')
Q2=[k for k in TEXT if k[0] in('answerai-colbert-small-v1','mLateOn','mxbai-edge-colbert-v0-17m','mxbai-edge-colbert-v0-32m','GTE-ModernColBERT-v1')]
cnt=sum((sim[k]['int4norm']>=0.95).astype(int) for k in Q2)
for k in Q2: print('  %-45s obs %.3f P(>=.95)=%.2f'%(k,obs(k,'int4norm'),(sim[k]['int4norm']>=0.95).mean()))
print('observed count',sum(obs(k,'int4norm')>=0.95 for k in Q2),'; P(count>=8)=%.2f; distribution'%(cnt>=8).mean(),np.bincount(cnt,minlength=11)/B)
print('\n== Q3 LateOn int4norm <0.9 and mean variant >=0.95')
for k in TEXT:
    if k[0]=='LateOn': print(' ',k,'obs int4norm %.4f P(<0.9)=%.2f ; int4norm(mean) obs %.4f P(>=.95)=%.2f'%(obs(k,'int4norm'),(sim[k]['int4norm']<0.9).mean(),obs(k,'int4norm(mean)'),(sim[k]['int4norm(mean)']>=0.95).mean()))
ks=[k for k in TEXT if k[0]=='LateOn']; print('  P(both <0.9)=%.2f'%((sim[ks[0]]['int4norm']<0.9)&(sim[ks[1]]['int4norm']<0.9)).mean())
print('\n== lower-bound margins for int4norm(mean): z=(point-1.96sd... ) approx P(fresh lower bound >=0.95)=Phi((point-0.95)/sd-1.96)')
def margin(ks,label):
    r=[]
    for k in ks:
        pt=obs(k,'int4norm(mean)'); sd=sim[k]['int4norm(mean)_sd']; r.append((norm.cdf((pt-0.95)/sd-1.96),k,pt,lo(k,'int4norm(mean)')))
    r.sort(key=lambda t:t[0]); print(label); 
    for p,k,pt,l in r[:4]: print('   %-45s point %.3f lower %.3f  approx P(pass)=%.3f'%(k,pt,l,p))
    return np.array([x[0] for x in r])
pt=margin(TEXT,'TEXT (Q4: need >=23 of 24)')
ps=margin(PAGE,'PAGE (Q8: need 14 of 14)'); 
print(' approx P(all 14 pass)=%.3f'%np.prod(ps))
# Q4 poisson-binomial P(>=23)
def pb(p,need):
    dist=np.array([1.0])
    for q in p: dist=np.convolve(dist,[1-q,q])
    return dist[need:].sum()
print(' approx P(>=23 of 24 text pass)=%.3f ; observed passes'%pb(pt,23),sum(lo(k,'int4norm(mean)')>=0.95 for k in TEXT))
pc=margin(SCI,'SCIDOCS (Q9c need 7 of 8)'); print(' approx P(>=7 of 8)=%.3f'%pb(pc,7))
print('Q9a int8 lower>=0.95 on SCI:',[round(lo(k,'int8(per_vector)'),3) for k in SCI])
print('\n== Q7 nemotron: int4, int4norm <0.5; mean-norm >=0.97 (point-based sim)')
for k in units:
    if k[0].startswith('nemotron'):
        print(' ',k[0][-6:],k[1][:6],'int4 %.3f int4norm %.3f P(int4norm<0.5)=%.2f int4norm(mean) %.3f P(>=.97)=%.2f'%(obs(k,'int4'),obs(k,'int4norm'),(sim[k]['int4norm']<0.5).mean(),obs(k,'int4norm(mean)'),(sim[k]['int4norm(mean)']>=0.97).mean()))
