import csv,json,glob,os,numpy as np
from scipy.stats import spearmanr
U=[{k:(float(v) if k not in('batch','model','ds') else v) for k,v in r.items()} for r in csv.DictReader(open('units.csv'))]
U=[u for u in U if not u['batch'].startswith('f13b')]
for u in U: u['nsr']=u['p2r']**2/588/(1-u['cosm']**2)
y=np.array([u['i4n']<0.95 for u in U])
def lomo(f,sign):
    errs=0;tp=fn=fp=0;preds=[]
    for m in {u['model'] for u in U}:
        tr=[u for u in U if u['model']!=m]; te=[u for u in U if u['model']==m]
        xs=np.array([sign*u[f] for u in tr]);ys=np.array([u['i4n']<0.95 for u in tr])
        best=None
        for t in np.unique(xs):
            for tt in [t]:
                acc=np.mean((xs>tt)==ys)
                if best is None or acc>best[0]: best=(acc,tt)
        # midpoint threshold
        s=np.sort(np.unique(xs));i=np.searchsorted(s,best[1]);thr=(s[i]+s[i+1])/2 if i+1<len(s) else s[i]
        for u in te: preds.append((sign*u[f]>thr, u['i4n']<0.95))
    p=np.array(preds);return np.mean(p[:,0]==p[:,1]),int(np.sum(p[:,0]&~p[:,1])),int(np.sum(~p[:,0]&p[:,1]))
print('leave-one-model-out (19 folds, 46 units; need=i4n<0.95 n=%d). acc, false-alarms, misses'%y.sum())
for f,s in [('nsr',1),('sd_int4',1),('top4',1),('top1',1),('cosm',1),('pe4',-1),('p2r',1)]:
    print(f,lomo(f,s))
H=[u for u in U if u['nsr']>0.5]
print('high-nsr subset n=%d: rho(nsr,i4n)=%.2f rho(top4,i4n)=%.2f rho(p2r,i4n)=%.2f rho(cosm,i4n)=%.2f rho(qcos,i4n)=%.2f rho(pe4,i4n)=%.2f rho(sd,i4n)=%.2f'%(len(H),*[spearmanr([u[f] for u in H],[u['i4n'] for u in H])[0] for f in['nsr','top4','p2r','cosm','qcos','pe4','sd_int4']]))
L=[u for u in U if u['nsr']<=0.5]
print('low-nsr subset n=%d: i4n min %.3f, # <0.95: %d, # <0.97: %d; m4n lowest %.3f; i4 <0.95: %d'%(len(L),min(u['i4n'] for u in L),sum(u['i4n']<.95 for u in L),sum(u['i4n']<.97 for u in L),min(u['m4n'] for u in L),sum(u['i4']<.95 for u in L)))
print('high subset i4n<0.95: %d of %d; m4n min %.3f'%(sum(u['i4n']<.95 for u in H),len(H),min(u['m4n'] for u in H)))
# fraction of the plain-int4 gap closed by norm
print('\nfraction of (m4n - i4) closed by norm, i4<0.95 units:')
for u in sorted(U,key=lambda u:u['nsr']):
    if u['i4']<0.95 and u['m4n']-u['i4']>0.05: print('%-34s %-10s nsr %.3f closed %.2f (i4 %.3f -> i4n %.3f, m4n %.3f)'%(u['model'][:34],u['ds'].split('_')[-3] if 'subsampled' in u['ds'] else u['ds'].split('_')[-1],u['nsr'],(u['i4n']-u['i4'])/(u['m4n']-u['i4']),u['i4'],u['i4n'],u['m4n']))
x=[(u['nsr'],(u['i4n']-u['i4'])/(u['m4n']-u['i4'])) for u in U if u['i4']<0.95 and u['m4n']-u['i4']>0.05]
print('n=%d rho(nsr, fraction closed)=%.2f'%(len(x),spearmanr(*zip(*x))[0]))
# per-query
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
print('\nper-query ndcg@10: share of queries with ndcg within 1e-9 of float32 / fraction with nonzero float32 / mean over queries with float32>0')
for f in sorted(glob.glob(R+'f11b*/*/int4norm/*.json')+glob.glob(R+'f9b*/lightonai--LateOn/int4norm/*.json')):
    d=json.load(open(f));p=d['per_query_ndcg@10'];b=np.array(p['float32'])
    nm=os.path.basename(f)[:-5].replace('nemotron-colembed-vl-','nem-').replace('_test_subsampled','')
    if 'nem' not in nm and 'LateOn' not in nm: continue
    out=[]
    for l in ['int4','int4norm','int4(mean)']:
        q=np.array(p[l]);pos=b>0
        out.append('%s: same %.2f, zero-when-b>0 %.2f, mean ratio(b>0) %.2f'%(l,np.mean(np.abs(q-b)<1e-9),np.mean((q==0)&pos),q[pos].mean()/b[pos].mean()))
    print(nm,'nq',len(b),'b>0 %.2f'%np.mean(b>0),'base nd@10 %.3f'%b.mean(),'|',' | '.join(out))
