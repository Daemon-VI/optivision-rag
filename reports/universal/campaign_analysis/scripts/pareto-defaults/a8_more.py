from common import *
from scipy import stats
import glob
m=get(); G=m[~m.legacy].copy()
# ---- dtype fix
from load import wide_all
w,t,bs=wide_all()
dt=bs.set_index('key').dtype.map(lambda s:'f16' if '16' in s else ('f32' if '32' in s else 'auto'))
G['dt']=(G.model+'|'+G.ds).map(dt)
print("pairs by baseline inference dtype:",G.groupby('dt').key.nunique().to_dict())
print("== float16-inference vs float32 (image only; confounded with model size/family) ==")
for l in ['binary','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)']:
    g=G[(G.label==l)&(G.modal=='image')]; a=g[g.dt=='f16'].ret; c=g[g.dt=='f32'].ret; u=stats.mannwhitneyu(a,c)
    print(f"{l:48s} f16 n={len(a)} median {a.median():.4f} | f32 n={len(c)} median {c.median():.4f} | p={u.pvalue:.2f}")
# ---- cluster bootstrap over models for coverage of top recipes
print("\n== cluster bootstrap over MODELS (resample models, keep both datasets) for coverage lo>=T ==")
rng=np.random.default_rng(3)
models=sorted(G.model.unique())
cands=['float16','int8(per_vector)','int4(mean)','hierarchical_merge(0.5)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.25) > binary']
rows=[]
for l in cands:
    g=G[G.label==l]
    for T in (.95,.97,.99):
        hit=g.assign(h=(g.lo>=T)).groupby('model').h.agg(['sum','count'])
        pt=hit['sum'].sum()/hit['count'].sum()
        bs_=[]
        for _ in range(2000):
            ms=rng.choice(hit.index,len(hit)); bs_.append(hit.loc[ms,'sum'].sum()/hit.loc[ms,'count'].sum())
        rows.append((l,T,pt,*np.percentile(bs_,[2.5,97.5]),g.comp.median()))
        print(f"{l:48s} T={T} coverage {pt:.3f} model-cluster boot CI [{np.percentile(bs_,2.5):.3f},{np.percentile(bs_,97.5):.3f}] ({len(hit)} models) comp {g.comp.median():.1f}x")
pd.DataFrame(rows,columns=['label','thr','cov','lo','hi','comp']).to_csv('cluster_boot.csv',index=False)
# ---- high-dim models: project(128) recipes
print("\n== dim>128 pairs: coverage and compression of project(128)-based pipelines ==")
H=G[G.dim>128]; print("pairs",H.key.nunique(),"models",H.model.nunique(), sorted(H.dim.unique()))
for l in ['int8(per_vector)','int4(mean)','binary','project(128)','project(128) > int8(per_vector)','project(128) > int4(mean)','project(128) > binary','hierarchical_merge(0.25) > project(128) > int8(per_vector)','hierarchical_merge(0.25) > project(128) > int4(mean)','hierarchical_merge(0.25) > project(128) > binary','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > binary']:
    g=H[H.label==l]
    if len(g)==0: continue
    print(f"{l:58s} n={len(g)} lo>=.95 {(g.lo>=.95).sum()} lo>=.97 {(g.lo>=.97).sum()} lo>=.99 {(g.lo>=.99).sum()}  comp median {g.comp.median():.1f} [{g.comp.min():.1f},{g.comp.max():.1f}] ret min {g.ret.min():.3f}")
# ---- pareto: usable front (only rows with lo>=.95) recurrence; and shipped pareto flag
print("\n== usable front: among rows with lo>=0.95, those not dominated in (comp up, lo up) ==")
cnt={}
for k,g in G.groupby('key'):
    g=g[g.lo>=.95].sort_values(['comp','lo'],ascending=[False,False]); best=-1
    for i,r in g.iterrows():
        if r.lo>best+1e-12: cnt.setdefault(r.label,set()).add(k); best=r.lo
U=pd.Series({l:len(v) for l,v in cnt.items()}).sort_values(ascending=False); print(U.head(14).to_string())
print("\n== shipped pareto flag (point retention, bytes) from wide/*.json: recurrence ==")
w['n']=1; pf=w[w.pareto].groupby('label').key.nunique().sort_values(ascending=False); print(pf.head(14).to_string()); print("mean front size",w[w.pareto].groupby('key').size().mean())
# front labels with high retention
print("\nof the shipped front rows, share with ret<0.9:", (w[w.pareto].ret<0.9).mean().round(3), "; front rows with lo>=0.95:",(w[w.pareto].lo>=.95).mean().round(3))
# ---- selection_rules aggregate
print("\n== per-model calibrated selection (selection_rules/*.json, 20 query splits, same corpus) vs fixed recipe ==")
REPO="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign"
rec=[]
for f in glob.glob(REPO+"/*/selection_rules/*.json"):
    d=json.load(open(f)); mm=f.replace("\\","/").split("/campaign/")[1].split("/")[0]
    for r in d['results']:
        rec.append(dict(model=mm,ds=d['dataset'],ref=r['reference'],rule=r['rule'],target=r['target'],met=r['met'],comp=r['compression_median'],sel=r['selected'],splits=r['splits'],hmean=r['holdout_mean'],hmin=r['holdout_min']))
S=pd.DataFrame(rec); print("files",len(glob.glob(REPO+"/*/selection_rules/*.json")),"rows",len(S))
q=S[S.ref=='labels'].groupby(['rule','target']).agg(files=('met','size'),met_mean=('met','mean'),share_all_met=('met',lambda x:(x>=0.999).mean()),met_p10=('met',lambda x:np.percentile(x,10)),comp_med=('comp','median'),hmean=('hmean','mean'))
print(q.round(3).to_string())
S.to_csv('selection_flat.csv',index=False)
