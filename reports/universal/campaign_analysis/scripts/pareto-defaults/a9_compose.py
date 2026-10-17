from common import *
from scipy import stats
m=get(); G=m[~m.legacy].copy()
P=G.pivot_table(index='key',columns='label',values='ret'); L=G.pivot_table(index='key',columns='label',values='lo')
info=G.groupby('key').agg(modal=('modal','first'),ds=('ds','first'),model=('model','first'),family=('family','first'),dim=('dim','first'))
NT=0
print("== A. Do token-merge and codec losses compose multiplicatively? ret(hm>codec) vs ret(hm)*ret(codec) ==")
for hm in ('hierarchical_merge(0.25)','hierarchical_merge(0.33)'):
    for cd in ('int8(per_vector)','int4(mean)','binary','lloyd2(7)'):
        a=P[f'{hm} > {cd}']; pred=P[hm]*P[cd]; d=(a-pred)
        rng=np.random.default_rng(0); ci=np.percentile([rng.choice(d.values,len(d)).mean() for _ in range(3000)],[2.5,97.5])
        rho=stats.spearmanr(1-P[hm],1-P[cd]).statistic; NT+=1
        print(f"{hm[-5:]} > {cd:18s} actual-pred mean {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] mean abs err {d.abs().mean():.4f}; mean ret actual {a.mean():.4f}; spearman(merge loss, codec loss across pairs)={rho:+.2f}")
print("\n   by modality (hm0.25 > binary): ")
for mo in ('image','text'):
    ix=info.index[info.modal==mo]; d=P.loc[ix,'hierarchical_merge(0.25) > binary']-P.loc[ix,'hierarchical_merge(0.25)']*P.loc[ix,'binary']; print(mo,f"{d.mean():+.4f}", f"n={len(ix)}")
print("\n== B. Fragility index: number of 10 headline recipes with lo<0.95, per model-dataset ==")
head=['int8(per_vector)','int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.5)','hierarchical_merge(0.25)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > lloyd2(7)','hierarchical_merge(0.25) > binary']
F=(L[head]<.95).sum(axis=1); info['frag']=F
print(info.groupby('ds').frag.describe().round(2).to_string())
print(info.sort_values('frag',ascending=False).head(14)[['frag','ds','dim']].to_string())
# consistency across the two datasets for same model (image)
im=info[info.modal=='image'].reset_index().pivot(index='model',columns='ds',values='frag').dropna()
r=stats.spearmanr(im.docvqa,im.infovqa); NT+=1; print("\nimage models: spearman(frag docvqa, frag infovqa) =%.2f p=%.1e n=%d"%(r.statistic,r.pvalue,len(im)))
# retention-loss correlation between datasets for binary / merges (model level)
for l in ['binary','hierarchical_merge(0.25)','hierarchical_merge(0.25) > int8(per_vector)','lloyd2(7)']:
    x=P[l].to_frame('r').join(info).query("modal=='image'").reset_index().pivot(index='model',columns='ds',values='r').dropna(); r=stats.spearmanr(x.docvqa,x.infovqa); NT+=1
    print(f"   {l:48s} spearman(ret docvqa, ret infovqa)={r.statistic:+.2f} p={r.pvalue:.3f} n={len(x)}")
# family grouping: ICC-like — variance share between families for binary retention (image only)
print("\n== C. how much of binary/merge retention variance is between-family (image, docvqa)? (eta^2) ==")
for l in ['binary','hierarchical_merge(0.25)','hierarchical_merge(0.25) > int8(per_vector)']:
    g=G[(G.label==l)&(G.ds=='docvqa')]
    grand=g.ret.mean(); ssb=sum(len(x)*(x.ret.mean()-grand)**2 for _,x in g.groupby('family')); sst=((g.ret-grand)**2).sum()
    # permutation p
    rng=np.random.default_rng(1); obs=ssb/sst; fam=g.family.values; r=g.ret.values; cnt=0
    for _ in range(3000):
        pf=rng.permutation(fam); s=0
        for f in set(pf): s+=(pf==f).sum()*(r[pf==f].mean()-grand)**2
        cnt+= s/sst>=obs
    NT+=1
    print(f"{l:48s} eta2={obs:.2f} perm p={(cnt+1)/3001:.3f} families={g.family.nunique()} n={len(g)}")
print("\nTESTS in this script:",NT)
