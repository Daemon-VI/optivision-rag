from common import *
m=get()
info=m.groupby('key').agg(modal=('modal','first'),ds=('ds','first'),model=('model','first'),family=('family','first'),dim=('dim','first'),base=('base','first'),vpd=('vpd','max'),legacy=('legacy','first'))
cands=['float16','int8(per_vector)','int4(mean)','int4','lloyd2(7)','binary','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.33) > int8(per_vector)','hierarchical_merge(0.33) > int4(mean)','hierarchical_merge(0.5)','hierarchical_merge(0.25)']
for c in cands:
    g=m[m.label==c]
    f=g[g.lo<.95].sort_values('lo')
    print(f"\n{c}: n={len(g)} fail lo<.95: {len(f)}  min lo={g.lo.min():.3f}")
    for _,r in f.iterrows(): print(f"   {r.key}  lo={r.lo:.3f} ret={r.ret:.3f} base={r.base} dim={r.dim}")
