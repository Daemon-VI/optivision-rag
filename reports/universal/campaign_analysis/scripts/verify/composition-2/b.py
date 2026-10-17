import pickle,numpy as np,collections
from scipy import stats
recs=pickle.load(open('recs.pkl','rb'))
text=lambda ds:'scifact' in ds.lower()
def permodel(cn,grp,f=lambda r:True):
    per=collections.defaultdict(list)
    for r in recs:
        if r[3]==cn and text(r[1])==(grp=='text') and f(r): per[r[0]].append(r[4]-r[5])
    return {k:np.mean(v) for k,v in per.items()}
fam=lambda k:('mxbai-edge' if 'edge' in k else 'LFM' if 'LFM' in k else 'LateOn' if 'LateOn' in k else 'colbertv2' if 'colbertv2' in k else k)
for cn in ('binary','int4','lloyd2','int4(mean)','int8'):
    t=permodel(cn,'text'); v=np.array(list(t.values()))
    loo=[np.delete(v,i).mean() for i in range(len(v))]
    print(cn,'LOO range %.4f..%.4f'%(min(loo),max(loo)),'sign neg %d/%d'%((v<0).sum(),len(v)),'Wilcoxon p=%.4f'%stats.wilcoxon(v).pvalue, 'median %.4f'%np.median(v))
    # family-collapsed
    fm=collections.defaultdict(list)
    for k,x in t.items(): fm[fam(k)].append(x)
    fv=np.array([np.mean(x) for x in fm.values()])
    rng=np.random.default_rng(1);bs=[rng.choice(fv,len(fv)).mean() for _ in range(10000)]
    print('   family-collapsed n=%d mean %.4f CI %s Wilcoxon p=%.4f'%(len(fv),fv.mean(),np.round(np.percentile(bs,[2.5,97.5]),4),stats.wilcoxon(fv).pvalue))
    im=np.array(list(permodel(cn,'image').values()))
    print('   image Wilcoxon p=%.4f neg %d/%d; text-vs-image MWU p=%.4f'%(stats.wilcoxon(im).pvalue,(im<0).sum(),len(im),stats.mannwhitneyu(v,im).pvalue))
# paired int4(mean)-int4 error
a=permodel('int4(mean)','text');b=permodel('int4','text')
d=np.array([a[k]-b[k] for k in a]);print('int4mean-int4 text',d.mean(),'median',np.median(d),stats.wilcoxon(d).pvalue)
d2=np.delete(d,np.argmax(d));print(' without top',d2.mean(),stats.wilcoxon(d2).pvalue)
# per ratio
for ra in (0.1,0.25,0.33,0.5):
    print(ra,{cn:(round(np.mean(list(permodel(cn,'text',lambda r:r[2]==ra).values())),4),round(np.mean(list(permodel(cn,'image',lambda r:r[2]==ra).values())),4)) for cn in ('binary','int4','lloyd2')})
# dependence on product level: text vs image mean merge and codec retention
for g in ('text','image'):
    for cn in ('binary','int4','lloyd2'):
        s=[r for r in recs if r[3]==cn and text(r[1])==(g=='text')]
        print(g,cn,'mean merge %.3f codec %.3f actual %.3f prod %.3f'%tuple(np.mean([r[i] for r in s]) for i in (6,7,4,5)))
# relative gap: (a-p)/(1-codec) fraction of the codec loss added
for g in ('text','image'):
    for cn in ('binary','int4','lloyd2'):
        s=[r for r in recs if r[3]==cn and text(r[1])==(g=='text')]
        print(g,cn,'mean codec-only loss %.4f, mean extra gap %.4f'%(np.mean([1-r[7] for r in s]),np.mean([r[5]-r[4] for r in s])))
