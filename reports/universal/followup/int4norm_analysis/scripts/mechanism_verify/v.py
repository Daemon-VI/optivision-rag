import json,glob,os,numpy as np
from scipy.stats import spearmanr
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
U=[]
for b in ['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs']:
    for f in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
        d=json.load(open(f)); f=f.replace(chr(92),'/'); p=f.split('/')
        model=p[-3]
        g=json.load(open(f.replace('/int4norm/','/geometry/')))
        r={x['label']:x for x in d['rows']}
        rt=lambda l:r[l]['retention:ndcg@5']
        n=d['norms']
        U.append(dict(model=model,ds=d['dataset'],nq=d['n_queries'],nd=d['n_docs'],i4=rt('int4'),i4n=rt('int4norm'),m4=rt('int4(mean)'),m4n=rt('int4norm(mean)'),
          sd=n['int4']['ratio_sd'],sdn=n['int4norm']['ratio_sd'],top1=g['second_moment_share']['top1'],top4=g['second_moment_share']['top4'],
          p2r=g['peak_to_rms']['mean'],cosm=g['doc_cos_to_mean']['mean'],pe4=g['codecs']['int4']['score_pearson']['mean'],f=f,
          pq=d['per_query_ndcg@10']))
print(len(U),len({u['model'] for u in U}))
from collections import Counter
print(Counter('pages' if 'pages' in u['f'] else 'scidocs' if 'scidocs' in u['f'] else 'text' for u in U))
for u in U: u['nsr']=u['p2r']**2/588/(1-u['cosm']**2)
A=lambda k:np.array([u[k] for u in U])
y=A('i4n')<0.95; print('need',y.sum())
def auc(x):
    return np.mean([(p>n)+.5*(p==n) for p in x[y] for n in x[~y]])
gain=A('i4n')-A('i4')
for k in ['nsr','sd','top1','top4','cosm','p2r','pe4']:
    x=A(k); print(k,[round(spearmanr(x,t)[0],2) for t in (A('i4n'),A('i4'),gain)],'auc',round(auc(x),3))
print('gainmean',[round(spearmanr(A(k),A('m4n')-A('m4'))[0],2) for k in ['top4','nsr','sd']])
H=A('nsr')>0.5;L=~H
print('H',H.sum(),'i4n<.95 in H',(A('i4n')[H]<.95).sum(),'m4n min H',A('m4n')[H].min(),'L',L.sum(),'L i4n min',A('i4n')[L].min(),(A('i4n')[L]<.95).sum(),'L m4n min',A('m4n')[L].min(),'L i4<.95',(A('i4')[L]<.95).sum())
print('L i4<.95 & i4n min',min(u['i4n'] for u in U if u['nsr']<=.5 and u['i4']<.95))
print([ (u['model'],u['ds'][-20:]) for u in U if u['nsr']<=.5 and u['i4n']<.95])
print('H models',Counter(u['model'] for u in U if u['nsr']>.5))
# fraction closed
S=[u for u in U if u['i4']<.95 and u['m4n']-u['i4']>.05]
fr=[(u['i4n']-u['i4'])/(u['m4n']-u['i4']) for u in S]
print('n',len(S),'rho',spearmanr([u['nsr'] for u in S],fr)[0])
for u,f_ in sorted(zip(S,fr),key=lambda t:t[0]['nsr']): print(u['model'][:30],u['ds'][-25:],round(u['nsr'],2),round(f_,2),round(u['i4'],3),round(u['i4n'],3))
# model level
import collections
M=collections.defaultdict(list)
for u in U:M[u['model']].append(u)
ks=list(M)
mm=lambda k:[np.mean([u[k] for u in M[m]]) for m in ks]
print('model lvl top4 vs i4n',spearmanr(mm('top4'),mm('i4n'))[0],'cosm vs gain',spearmanr(mm('cosm'),[a-b for a,b in zip(mm('i4n'),mm('i4'))])[0])
# nemotron
for u in U:
    if 'nemotron' in u['model']: print(u['model'],u['ds'][-30:],'nsr',round(u['nsr'],1),round(u['i4n'],3),round(u['i4'],3),round(u['m4n'],3),'top4',round(u['top4'],2),'p2r',round(u['p2r'],1),'cosm',round(u['cosm'],3),'pe4',round(u['pe4'],2),'nq',u['nq'],u['nd'])
for u in U:
    if 'nemotron' in u['model'] or 'LateOn' in u['model'] and 'mLate' not in u['model']:
        b=np.array(u['pq']['float32']);out=[]
        for l in ['int4','int4norm']:
            q=np.array(u['pq'][l]);pos=b>0
            out.append('%s same %.2f z %.2f mr %.2f'%(l,np.mean(abs(q-b)<1e-9),np.mean((q==0)&pos),q[pos].mean()/b[pos].mean()))
        print(u['model'][-20:],u['ds'][-25:],'b>0 %.2f'%np.mean(b>0),out)
print('LateOn sd',[ (u['model'][-12:],round(u['sd'],4),round(u['sdn'],5),round(u['nsr'],2)) for u in U if 'LateOn' in u['model'] or 'nemotron' in u['model']][:20])
print('max sdn',A('sdn').max(), 'median sd',np.median(A('sd')))
for u in U:
    if 'colpali-v1.3' in u['model']: print(u['ds'],u['i4'],u['i4n'],u['nsr'])
