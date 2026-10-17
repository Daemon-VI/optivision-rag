from common import *
from scipy import stats
m=get(); G=m[~m.legacy].copy()
info=G.groupby('key').agg(modal=('modal','first'),ds=('ds','first'),model=('model','first'),family=('family','first'),dim=('dim','first'),base=('base','first'),vpd=('vpd','max')).reset_index()
NT=0
def boot_ci(x,f=np.mean,n=5000,seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x); return np.percentile([f(rng.choice(x,len(x))) for _ in range(n)],[2.5,97.5])
# ---- 1. docvqa vs infovqa paired by model (image only)
im=G[G.modal=='image'].pivot_table(index=['model','label'],columns='ds',values=['ret','lo'])
im=im.dropna()
labs=['float16','int8(per_vector)','int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.5)','hierarchical_merge(0.25)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > binary']
print("== 1. DocVQA minus InfoVQA retention, paired by model (image models with both) ==")
rows=[]
for l in labs:
    x=im.xs(l,level='label'); d=(x[('ret','docvqa')]-x[('ret','infovqa')]).values
    if len(d)<5: continue
    p=stats.wilcoxon(d).pvalue; NT+=1
    lo,hi=boot_ci(d)
    print(f"{l:48s} n={len(d)} mean diff {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] docvqa<infovqa in {(d<0).sum()}/{len(d)}, wilcoxon p={p:.1e}")
b=G.groupby('key').base.first(); bb=pd.DataFrame({'base':b}).join(info.set_index('key')[['ds','model']])
x=bb[bb.ds!='scifact'].pivot(index='model',columns='ds',values='base').dropna(); print("baseline nDCG@5 docvqa median %.3f infovqa median %.3f"%(x.docvqa.median(),x.infovqa.median()))
# ---- 2. is it baseline ceiling? within-dataset spearman retention vs baseline across models
print("\n== 2. Spearman(retention, baseline nDCG@5) within dataset (model-level) ==")
for l in ['binary','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int8(per_vector)','lloyd2(7)']:
    for ds in ('docvqa','infovqa','scifact'):
        g=G[(G.label==l)&(G.ds==ds)]; r=stats.spearmanr(g.base,g.ret); NT+=1
        print(f"{l:44s}{ds:8s} n={len(g)} rho={r.statistic:+.2f} p={r.pvalue:.2f}")
# ---- 3. int4 vs int4(mean)
print("\n== 3. int4 (uncentred) vs int4(mean) (centred) ==")
a=G[G.label=='int4'].set_index('key'); c=G[G.label=='int4(mean)'].set_index('key'); j=a.join(c,lsuffix='_p',rsuffix='_c')
d=j.ret_c-j.ret_p
print(f"n={len(j)} mean gain from centring {d.mean():+.4f}, median {d.median():+.5f}, centred > plain in {(d>0).sum()}, equal-ish(|d|<.003) {(d.abs()<.003).sum()}")
print("plain int4 lo<0.95 in",(j.lo_p<.95).sum(),"pairs; centred lo<0.95 in",(j.lo_c<.95).sum())
print(j[j.lo_p<.95][['dim_p','modal_p','ret_p','ret_c','lo_p','lo_c']].round(3).to_string())
print("dim>=2048:",j[j.dim_p>=2048][['ret_p','ret_c']].round(3).to_dict('list'))
# ---- 4. binary vs native dim (image only to cut modality), docvqa only
print("\n== 4. native dim effect (image, binary & int4 plain) ==")
for l in ['binary','int4','lloyd2(7)']:
    for ds in ('docvqa','infovqa'):
        g=G[(G.label==l)&(G.ds==ds)]
        hi=g[g.dim>128].ret; lo=g[g.dim<=128].ret
        u=stats.mannwhitneyu(hi,lo); NT+=1
        print(f"{l:12s}{ds:8s} dim>128 n={len(hi)} median ret {hi.median():.4f} | dim=128 n={len(lo)} median {lo.median():.4f} | MWU p={u.pvalue:.3f}")
# ---- 5. merge and vpd: text vs image, retention of hm(0.25) vs vpd (confounded)
print("\n== 5. token merge hm(0.25) retention vs vectors/page ==")
g=G[G.label=='hierarchical_merge(0.25)']
for sub in ('image','text'):
    gg=g[g.modal==sub]; r=stats.spearmanr(gg.vpd,gg.ret); NT+=1
    print(sub,'n=',len(gg),'median ret %.3f'%gg.ret.median(),'vpd range %.0f-%.0f'%(gg.vpd.min(),gg.vpd.max()),'spearman(vpd,ret)=%.2f p=%.3f'%(r.statistic,r.pvalue))
print("image vs text hm(0.25) ret: image median %.4f text median %.4f MWU p=%.1e"%(g[g.modal=='image'].ret.median(),g[g.modal=='text'].ret.median(),stats.mannwhitneyu(g[g.modal=='image'].ret,g[g.modal=='text'].ret).pvalue)); NT+=1
# vpd vs ret binary within text
# ---- 6. text vs image failure rate per label with Fisher exact (dataset-confounded)
print("\n== 6. P(lo>=0.95) text vs image, Fisher exact (fully confounded with dataset) ==")
for l in ['int4(mean)','lloyd2(7)','binary','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.33) > int8(per_vector)']:
    g=G[G.label==l]; t=g[g.modal=='text']; i=g[g.modal=='image']
    tab=[[int((i.lo>=.95).sum()),int((i.lo<.95).sum())],[int((t.lo>=.95).sum()),int((t.lo<.95).sum())]]
    p=stats.fisher_exact(tab)[1]; NT+=1
    print(f"{l:48s} image {tab[0][0]}/{len(i)} text {tab[1][0]}/{len(t)} p={p:.1e}")
# ---- 7. docvqa vs infovqa within image: binary fail rate (74 image pairs)
print("\n== 7. image: P(lo>=.95) docvqa vs infovqa ==")
for l in ['binary','lloyd2(7)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > project(128) > binary']:
    g=G[(G.label==l)&(G.modal=='image')]; a=g[g.ds=='docvqa']; b2=g[g.ds=='infovqa']
    tab=[[int((a.lo>=.95).sum()),int((a.lo<.95).sum())],[int((b2.lo>=.95).sum()),int((b2.lo<.95).sum())]]; p=stats.fisher_exact(tab)[1]; NT+=1
    print(f"{l:52s} docvqa {tab[0][0]}/{len(a)} infovqa {tab[1][0]}/{len(b2)} p={p:.1e}")
# ---- 8. float16-inference baseline effect
print("\n== 8. baseline inference dtype (float16 vs float32) on retention ==")
from load import wide_all
w,t,bs=wide_all(); dt=bs.set_index('key').dtype.map(lambda s:'f16' if '16' in s else ('f32' if '32' in s else 'text/auto')); 
G['dt']=G.key.map(dt)
print(G.groupby(['dt']).key.nunique().to_dict())
for l in ['binary','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int8(per_vector)']:
    g=G[(G.label==l)&(G.modal=='image')]; a=g[g.dt=='f16'].ret; c=g[g.dt=='f32'].ret; u=stats.mannwhitneyu(a,c); NT+=1
    print(f"{l:48s} image f16 n={len(a)} median {a.median():.4f} | f32 n={len(c)} median {c.median():.4f} | MWU p={u.pvalue:.2f}")
print("\nTESTS RUN in this script:",NT)
