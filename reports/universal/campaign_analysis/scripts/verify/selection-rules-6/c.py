import json,glob,collections,numpy as np,os,re
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
fs=sorted(glob.glob(R+'*/selection_rules/*.json'))
rng=np.random.default_rng(0)
rows=[]
for f in fs:
    d=json.load(open(f)); model=os.path.basename(os.path.dirname(os.path.dirname(f)))
    ds='docvqa' if 'docvqa' in f else ('infovqa' if 'infovqa' in f else 'scifact')
    by={}
    for x in d['results']: by[(x['rule'],x['target'],x['reference'])]=x
    for (rule,t,ref),x in by.items():
        if ref!='baseline@1': continue
        l=by.get((rule,t,'labels'))
        if l is None: continue
        rows.append(dict(model=model,ds=ds,rule=rule,t=t,cb=x['compression_median'],cl=l['compression_median'],nb=x['selected'],nl=l['selected']))
print(len(rows), collections.Counter(r['ds'] for r in rows if r['rule']=='point'))
def gm(a): return float(np.exp(np.mean(np.log(a))))
for rule in ['calibrate() default','lower 0.999','point','bonferroni 0.975','lower 0.975','lower 0.99']:
    rr=[r for r in rows if r['rule']==rule]
    ok=[r for r in rr if r['cb'] is not None and r['cl'] is not None]
    # missing count
    print(rule,len(rr),len(ok), 'None b/l',sum(r['cb'] is None for r in rr),sum(r['cl'] is None for r in rr))
    lr=np.array([np.log(r['cb']/r['cl']) for r in ok])
    mods=sorted(set(r['model'] for r in ok))
    idx={m:[i for i,r in enumerate(ok) if r['model']==m] for m in mods}
    def est(sel): return np.exp(np.mean(lr[sel]))
    bs=[]
    for _ in range(4000):
        ms=rng.choice(len(mods),len(mods))
        sel=np.concatenate([idx[mods[i]] for i in ms]).astype(int)
        bs.append(np.mean(lr[sel]))
    ci=np.exp(np.percentile(bs,[2.5,97.5]))
    print('  ratio gm %.3f CI [%.3f,%.3f] lower-share %.2f  median ratio %.3f  pooled gm b %.2f l %.2f'%(np.exp(lr.mean()),*ci,(lr<0).mean(),np.exp(np.median(lr)),gm([r['cb'] for r in ok]),gm([r['cl'] for r in ok])))
    # by dataset
    for ds in ['docvqa','infovqa','scifact']:
        s=[i for i,r in enumerate(ok) if r['ds']==ds]
        if s: print('   ',ds,len(s),'%.3f'%np.exp(lr[s].mean()))
    # leave-one-model-out range
    lo=[np.exp(np.mean(lr[[i for i in range(len(ok)) if ok[i]['model']!=m]])) for m in mods]
    print('   LOMO range %.3f-%.3f'%(min(lo),max(lo)), 'by target',{t:round(float(np.exp(np.mean([lr[i] for i,r in enumerate(ok) if r['t']==t]))),3) for t in (0.95,0.97,0.99)})
    # cells where ratio==1
    print('   ratio==1 share %.2f'%np.mean(np.abs(lr)<1e-9))
