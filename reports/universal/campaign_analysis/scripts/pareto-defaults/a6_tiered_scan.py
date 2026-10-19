from load import *
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
w,t,b=wide_all()
b['modal']=np.where(b.ds=='scifact','text','image')
t['modal']=np.where(t.ds=='scifact','text','image'); w['modal']=np.where(w.ds=='scifact','text','image')
print("pairs",b.key.nunique(),b.modal.value_counts().to_dict(),"baseline dtype:",b.dtype.value_counts().to_dict(), "src",b.src.value_counts().to_dict())
print("\n== TIERED (binary/ward>binary hot in RAM, int8 all-vectors cold on disk, 50 candidates) ==")
t['hotname']=t.hot
for h,g in t.groupby('hot'):
    print(f"\nhot={h}: n={len(g)}  ram compression median {g.ramc.median():.1f}x (min {g.ramc.min():.1f}, max {g.ramc.max():.1f}); disk compression median {g.diskc.median():.2f}x")
    for T in (.95,.97,.99):
        for sub in ('all','image','text'):
            gg=g if sub=='all' else g[g.modal==sub]
            k=int((gg.lo>=T).sum()); print(f"   lo>={T} {sub}: {k}/{len(gg)}",end=';')
        print()
    print(f"   retention median {g.ret.median():.4f}, min {g.ret.min():.3f}, share ret>=1: {(g.ret>=1).mean():.2f}")
    print("   hot ms median %.1f rescore ms median %.2f"%(g.hotms.median(),g.resms.median()))
# compare tiered vs untiered hot index alone (same RAM bytes)
cmp=[]
for h,lab in (('binary','binary'),('ward 1/4 > binary','hierarchical_merge(0.25) > binary')):
    a=t[t.hot==h].set_index('key'); c=w[w.label==lab].set_index('key')
    j=a.join(c,lsuffix='_t',rsuffix='_u',how='inner')
    d=j.ret_t-j.ret_u
    rng=np.random.default_rng(0); bs=[np.mean(rng.choice(d.values,len(d))) for _ in range(5000)]
    wil=stats.wilcoxon(d.values)
    print(f"\n{h}: tiered minus untiered retention (same RAM): mean {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] median {d.median():+.4f} n={len(d)} wilcoxon p={wil.pvalue:.2e}")
    for T in (.95,.97,.99):
        print(f"   lo>={T}: untiered {(j.lo_u>=T).sum()}/{len(j)} -> tiered {(j.lo_t>=T).sum()}/{len(j)}")
    # by dataset
    for ds,gg in j.groupby('ds_t'):
        dd=(gg.ret_t-gg.ret_u); print(f"   {ds}: mean gain {dd.mean():+.4f} (n={len(dd)}), untiered mean ret {gg.ret_u.mean():.3f} tiered {gg.ret_t.mean():.3f}")
    cmp.append(j.assign(hot_=h))
# is tiered gain predicted by the untiered shortfall? 
j=cmp[0]; print("\nbinary: spearman(untiered ret, tiered gain)=",stats.spearmanr(j.ret_u,j.ret_t-j.ret_u))
# 2-tier at scifact: candidates 50/5183 docs =1% vs 10% for image
print("\ncandidate share: image 50/500=10%; scifact 50/5183=0.96%")
print("scifact tiered vs image tiered ret (binary hot):",t[(t.hot=='binary')].groupby('modal').ret.agg(['mean','min','count']).round(3).to_dict())
# == SCAN LATENCY ==
print("\n== SCAN LATENCY (maxsim on decoded vectors; compression of the codec does not reduce this) ==")
f32=w[w.label=='float32'].set_index('key')
w=w.join(f32.scan.rename('scan32'),on='key'); w['rel_scan']=w.scan/w.scan32; w['rel_vpd']=w.vpd/w.join(f32.vpd.rename('vpd32'),on='key').vpd32
w['rel_dim']=w.dim/w.join(f32.dim.rename('dim32'),on='key').dim32
w['rel_bytes']=1/w.comp
grp=w.groupby('group').agg(rel_scan=('rel_scan','median'),rel_bytes=('rel_bytes','median'),rel_vpd=('rel_vpd','median'),rel_dim=('rel_dim','median'),n=('key','size'))
print(grp.round(3).to_string())
# log-log regression rel_scan ~ rel_vpd + rel_dim + codec-only bits
x=np.column_stack([np.ones(len(w)),np.log(w.rel_vpd),np.log(w.rel_dim)]); y=np.log(w.rel_scan)
ok=np.isfinite(y)&np.isfinite(x).all(1); beta,res,_,_=np.linalg.lstsq(x[ok],y[ok],rcond=None)
pred=x[ok]@beta; r2=1-((y[ok]-pred)**2).sum()/((y[ok]-y[ok].mean())**2).sum()
# bootstrap by pair
keys=w.key.unique(); rng=np.random.default_rng(1); bs=[]
wk={k:i for k,i in w.groupby('key').indices.items()}
for _ in range(500):
    ks=rng.choice(keys,len(keys)); idx=np.concatenate([wk[k] for k in ks]); xx=x[idx];yy=y[idx];o=np.isfinite(yy)&np.isfinite(xx).all(1)
    bs.append(np.linalg.lstsq(xx[o],yy[o],rcond=None)[0])
bs=np.array(bs)
print(f"\nlog(rel_scan) = a + {beta[1]:.3f}*log(rel_vpd) + {beta[2]:.3f}*log(rel_dim)  R2={r2:.3f}; bootstrap-over-pairs CI vpd [{np.percentile(bs[:,1],2.5):.3f},{np.percentile(bs[:,1],97.5):.3f}] dim [{np.percentile(bs[:,2],2.5):.3f},{np.percentile(bs[:,2],97.5):.3f}]")
# codec-only rows: scan relative
c=w[w.group=='codec']; print("codec-only rel_scan median by label:"); print(c.groupby('label').rel_scan.median().round(3).to_string())
# bytes vs scan correlation
print("\nspearman(compression, 1/rel_scan) all rows:",stats.spearmanr(w.comp,1/w.rel_scan,nan_policy='omit'))
print("spearman among token-merge only rows:",stats.spearmanr(w[w.group.str.startswith('token')].comp,1/w[w.group.str.startswith('token')].rel_scan,nan_policy='omit'))
# the interesting: speed-up for robust defaults
for lab in ['float16','int8(per_vector)','int4(mean)','binary','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.33) > int8(per_vector)']:
    g=w[w.label==lab]; print(f"{lab:48s} n={len(g)} rel_scan median {g.rel_scan.median():.3f} IQR [{g.rel_scan.quantile(.25):.3f},{g.rel_scan.quantile(.75):.3f}]  compression median {g.comp.median():.1f}x")
# absolute scan versus scale
print("\nbaseline scan ms vs vpd*n_docs: spearman", stats.spearmanr(b.scan,b.vpd*b.nd))
print(b.groupby('ds').scan.median())
