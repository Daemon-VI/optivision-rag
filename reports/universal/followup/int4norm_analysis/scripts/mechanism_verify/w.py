exec(open('v.py').read().split("print('H',H.sum()")[0].split("print(len(U)")[0])
for u in U: u['nsr']=u['p2r']**2/588/(1-u['cosm']**2)
y=np.array([u['i4n']<0.95 for u in U]); models=sorted({u['model'] for u in U})
def best_thr(x,yy):
    s=np.unique(x);b=(-1,None)
    for t in s:
        a=np.mean((x>t)==yy)
        if a>b[0]:b=(a,t)
    i=np.searchsorted(s,b[1]);thr=(s[i]+s[i+1])/2 if i+1<len(s) else s[i]
    return b[0],thr
for k,sg in [('nsr',1),('sd',1),('top4',1),('top1',1),('cosm',1),('pe4',-1),('p2r',1)]:
    x=np.array([sg*u[k] for u in U]); ia,thr=best_thr(x,y)
    pr=np.zeros(len(U),bool)
    for m in models:
        te=np.array([u['model']==m for u in U]); _,t=best_thr(x[~te],y[~te]); pr[te]=x[te]>t
    print(k,'insample',round(ia,3),'thr',round(sg*thr,4),'lomo acc',round(np.mean(pr==y),3),'FA',int((pr&~y).sum()),'miss',int((~pr&y).sum()))
print('low nsr with i4<.95:',sorted(round(u['i4n'],3) for u in U if u['nsr']<=.5 and u['i4']<.95))
print('mxbai17 nfcorpus & mLateOn scifact i4n', [(u['ds'][-12:],u['i4n']) for u in U if 'mLateOn' in u['model'] and 'scifact' in u['ds']])
print('sdn all<=?',sorted(u['sdn'] for u in U)[-3:], 'sd by model:',{m[-22:]:round(np.mean([u['sd'] for u in U if u['model']==m]),4) for m in models if 'EVIE' in m or 'ultron' in m or 'EVIE' in m.upper()})
print('answerai nfc sdn mean',[ (u['sdn']) for u in U if 'answer' in u['model'] and 'nfc' in u['ds']])
