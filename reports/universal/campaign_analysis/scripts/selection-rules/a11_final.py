import sys; sys.path.insert(0,'.')
from load import *
U='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign'
a=json.load(open(U+'/batches/11_text/orphans/colbertv2.0/selection_rules/colbertv2.0_scifact.json')); b=json.load(open(U+'/lightonai--colbertv2.0/selection_rules/colbertv2.0_scifact.json'))
ka={(r['reference'],r['rule'],r['target']):[o['selected'] for o in r['outcomes']] for r in a['results']}
kb={(r['reference'],r['rule'],r['target']):[o['selected'] for o in r['outcomes']] for r in b['results']}
same=np.mean([x==y for k in ka for x,y in zip(ka[k],kb[k])]); print('orphan vs lightonai colbertv2.0 same choice share',same, 'cand retentions equal',all(abs(x['retention_labels_all']-y['retention_labels_all'])<1e-9 for x,y in zip(a['candidates'],b['candidates'])), 'maxdiff',max(abs(x['retention_labels_all']-y['retention_labels_all']) for x,y in zip(a['candidates'],b['candidates'])))
# merge vs none fail ratio, cluster bootstrap
O=pd.read_pickle(OUT+'/O.pkl'); rng=np.random.default_rng(5)
L=O[(O.reference=='labels')&O.sel_ok].copy(); L['fail']=(L.hold<L.target).astype(float)
L['merged']=L.sel.str.contains('merge')
for r in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.975']:
    x=L[L.rule==r]; models=x.model.unique(); gp={m:x[x.model==m] for m in models}
    diffs=[]
    for _ in range(500):
        xx=pd.concat([gp[m] for m in rng.choice(models,len(models))])
        diffs.append(xx[xx.merged].fail.mean()-xx[~xx.merged].fail.mean())
    print(r,'fail merged',x[x.merged].fail.mean().round(4),'n',x.merged.sum(),'fail codec-only',x[~x.merged].fail.mean().round(4),'n',(~x.merged).sum(),'diff CI',np.percentile(diffs,[2.5,97.5]).round(4))
# per-target best compression subject to pooled met>=0.95 with model-cluster lower CI
T=pd.read_csv(OUT+'/rule_table_labels.csv')
T['ok']=T.met_lo>=0.95
print(T[T.target!='all'][['rule','target','met','met_lo','gm_comp_sel','ok']].round(3).to_string())
