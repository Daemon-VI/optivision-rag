import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy import stats
rng=np.random.default_rng(0)
U=load(); res=[]
def boot(x,B=10000):
    n=len(x); idx=rng.integers(0,n,(B,n)); m=x[idx].mean(1); return np.percentile(m,[2.5,97.5])
print('%-40s %-5s %-9s %5s %7s %6s %6s %6s %-17s %7s'%('unit','pair','','nq','mean','nchg','up','down','boot95','signP'))
for k,u in U.items():
    pq=u['d']['per_query_ndcg@10']
    for a,b,tag in [('int4norm','int4','P'),('int4norm(mean)','int4(mean)','M')]:
        x=np.array(pq[a])-np.array(pq[b]); n=len(x)
        up=(x>1e-9).sum(); dn=(x<-1e-9).sum(); ch=up+dn
        ci=boot(x); p=stats.binomtest(int(up),int(ch),0.5).pvalue if ch>0 else 1.0
        # check match with row
        rr=row(u['d'],a)['ndcg@10']-row(u['d'],b)['ndcg@10']
        res.append((k,tag,n,x.mean(),ch,up,dn,ci,p,rr))
        print('%-40s %-5s %5d %+.4f %5d %6d %6d  [%+.4f,%+.4f] %.3f  rowdiff %+.4f'%(k[0][:22]+'/'+k[1][:10],tag,n,x.mean(),ch,up,dn,ci[0],ci[1],p,rr))
import pickle; pickle.dump(res,open(os.path.dirname(os.path.abspath(__file__))+'/b_res.pkl','wb'))
print()
for tag in 'PM':
    r=[x for x in res if x[1]==tag]
    neg=[x for x in r if x[3]<0]
    signeg=[x for x in r if x[7][1]<0]; sigpos=[x for x in r if x[7][0]>0]
    print(tag,'units',len(r),'mean<0:',len(neg),'CI entirely <0:',len(signeg),[ (x[0][0],x[0][1][:8]) for x in signeg],'CI entirely >0:',len(sigpos))
    print('   sign test p<0.05 & mean<0:',[(x[0][0],x[0][1][:8],round(x[8],3)) for x in neg if x[8]<0.05])
    print('   expected false-neg-significant by chance at 2.5%% one-sided: %.1f of %d'%(0.025*len(r),len(r)))
# noise floor: float16 vs float32 and int8 vs float32 changed queries
print('\nNoise floor: queries whose ndcg@10 differs from float32')
for k,u in U.items():
    pq=u['d']['per_query_ndcg@10']; f=np.array(pq['float32'])
    s=[]
    for l in ['float16','int8(per_vector)','int4','int4norm','int4(mean)','int4norm(mean)']:
        x=np.array(pq[l])-f; s.append('%s:%d'%(l[:9],(abs(x)>1e-9).sum()))
    print('%-40s nq=%d '%(k[0][:22]+'/'+k[1][:10],len(f))+' '.join(s))
