import csv,numpy as np
from scipy.stats import spearmanr
U=[{k:(float(v) if v not in('',) and k not in('batch','model','ds') else v) for k,v in r.items()} for r in csv.DictReader(open('units.csv'))]
U=[u for u in U if not u['batch'].startswith('f13b')]  # 46 units
print('n units',len(U),'models',len({u['model'] for u in U}))
A=lambda k:np.array([u[k] for u in U])
gain=A('i4n')-A('i4'); gainm=A('m4n')-A('m4')
nsr=A('p2r')**2/588/(1-A('cosm')**2)
feat=dict(sd_int4=A('sd_int4'),top1=A('top1'),top4=A('top4'),p2r=A('p2r'),cosm=A('cosm'),qcos=A('qcos'),maxabs=A('maxabs'),
 cos4=A('cos4'),pe4=A('pe4'),pe4m=A('pe4m'),one_m_cos4=1-A('cos4'),nsr=nsr,p2r_over_sqrt1mc=A('p2r')/np.sqrt(1-A('cosm')**2),
 sd_over_1mc2=A('sd_int4')/(1-A('cosm')**2),dim=A('dim'),p2rc=A('p2rc'),p2r_ratio=A('p2r')/A('p2rc'),mean_int4=A('mean_int4'))
tgt=dict(i4=A('i4'),i4n=A('i4n'),i4n_lo=A('i4n_lo'),gain=gain,gain_mean=gainm,m4n=A('m4n'),m4n_minus_i4n=A('m4n')-A('i4n'))
print('\nSpearman (n=46 units) feature vs target')
print('%-18s'%'',*['%12s'%t for t in tgt])
for f,x in feat.items():
    print('%-18s'%f,*['%12.2f'%spearmanr(x,y)[0] for y in tgt.values()])
# classification: needs centring = i4n point <0.95 ; AUC
def auc(x,lab):
    pos=x[lab];neg=x[~lab]
    return np.mean([(p>n)+0.5*(p==n) for p in pos for n in neg])
need=A('i4n')<0.95
need2=A('i4n_lo')<0.95
print('\nneeds centring (i4n point<0.95): n=%d of 46; (i4n lower bound <0.95): n=%d'%(need.sum(),need2.sum()))
print('plain int4 point<0.95: n=%d'%(A('i4')<0.95).sum())
for f,x in feat.items():
    a=auc(x,need); 
    # best threshold
    best=max(((max(np.mean((x>t)==need),np.mean((x<=t)==need)),t) for t in np.unique(x)))
    print('%-18s AUC(need centring)=%.3f  best single-threshold acc=%.3f @%.4g'%(f,a,best[0],best[1]))
# unit table of those flagged
print('\nunits with i4n<0.95 or i4<0.95:')
for u in sorted(U,key=lambda u:u['i4n']):
    if u['i4']<0.95 or u['i4n']<0.97: print('%-40s %-14s i4 %.3f i4n %.3f m4n %.3f top4 %.2f p2r %.1f cosm %.3f qcos %.2f sd %.4f nsr %.3f'%(u['model'][:40],u['ds'].split('_')[-1][:12] if 'docvqa' not in u['ds'] and 'tabf' not in u['ds'] else u['ds'].split('_')[1] if False else u['ds'][-30:-16],u['i4'],u['i4n'],u['m4n'],u['top4'],u['p2r'],u['cosm'],u['qcos'],u['sd_int4'],nsr[U.index(u)]))
# model-level (mean over units) spearman
import collections
M=collections.defaultdict(list)
for u in U:M[u['model']].append(u)
mm={k:{f:np.mean([u[f] for u in v]) for f in ['i4','i4n','m4n','top4','p2r','cosm','sd_int4','qcos','cos4','pe4']} for k,v in M.items()}
print('\nmodel-level n=%d'%len(mm))
for f in ['sd_int4','top4','p2r','cosm','qcos','cos4','pe4']:
    x=[m[f] for m in mm.values()]
    print(f,'rho vs i4n %.2f  vs gain %.2f'%(spearmanr(x,[m['i4n'] for m in mm.values()])[0],spearmanr(x,[m['i4n']-m['i4'] for m in mm.values()])[0]))
# sd vs i4 (Q5 replicate) text units
T=[u for u in U if u['batch'].startswith('f9b')]
print('Q5 replicate 24 text units rho(sd,i4)=%.3f'%spearmanr([u['sd_int4'] for u in T],[u['i4'] for u in T])[0])
