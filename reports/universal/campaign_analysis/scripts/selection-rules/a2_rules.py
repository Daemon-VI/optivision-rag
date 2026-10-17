import sys; sys.path.insert(0,'.')
from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
O=pd.read_pickle(OUT+'/O.pkl'); F=pd.read_pickle(OUT+'/F.pkl')
RULES=['point','lower 0.975','lower 0.99','lower 0.999','bonferroni 0.975','lower 0.975 + margin 0.01','calibrate() default']
rng=np.random.default_rng(0)
O['logc']=np.where(O.sel_ok,np.log(O.comp),0.0)   # abstain => float corpus, 1x (strictly <=1: counts as met)
O['met_abst1']=np.where(O.sel_ok,O.met,1.0)
def summ(d):
    s=d[d.sel_ok]
    return dict(n=len(d),abst=1-len(s)/len(d),met=s.met.mean(),met_abst1=d.met_abst1.mean(),
                gm_comp_sel=np.exp(s.logc.mean()) if len(s) else np.nan,gm_comp_all=np.exp(d.logc.mean()),med_comp=s.comp.median() if len(s) else np.nan)
def table(ref,by_extra=None):
    d=O[O.reference==ref]
    rows=[]
    for rule in RULES:
        for tgt in [None,0.99,0.97,0.95]:
            x=d[(d.rule==rule)&((d.target==tgt) if tgt else True)]
            r=summ(x); r.update(rule=rule,target=tgt if tgt else 'all')
            # model-cluster bootstrap CI (cells weighted by splits)
            models=x.model.unique(); g={m:x[x.model==m] for m in models}
            bs=[]
            for b in range(400):
                pick=rng.choice(models,len(models))
                xx=pd.concat([g[m] for m in pick]); s=xx[xx.sel_ok]
                bs.append((s.met.mean(),np.exp(s.logc.mean())))
            bs=np.array(bs); r['met_lo'],r['met_hi']=np.nanpercentile(bs[:,0],[2.5,97.5]); r['gm_lo'],r['gm_hi']=np.nanpercentile(bs[:,1],[2.5,97.5])
            # per-cell
            cm=x[x.sel_ok].groupby(['model','dataset','target']).met.mean()
            r['cells']=len(cm); r['cells_ge95']=(cm>=0.95).mean(); r['cells_lt80']=(cm<0.8).mean()
            rows.append(r)
    return pd.DataFrame(rows)
for ref in ['labels','baseline@1']:
    T=table(ref); T.to_csv(f"{OUT}/rule_table_{ref.replace('@','')}.csv",index=False)
    print('=== reference',ref)
    print(T[['rule','target','n','abst','met','met_lo','met_hi','met_abst1','cells','cells_ge95','cells_lt80','gm_comp_sel','gm_lo','gm_hi','gm_comp_all','med_comp']].round(3).to_string())
