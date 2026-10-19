import numpy as np, pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
rng=np.random.default_rng(3)
df=table()
df=df[df.codec.isin(CODECS)]
w=pd.read_csv('wide_ret.csv')
# bytes order (smaller better)
b=df.pivot_table(index=['model','dataset'],columns='codec',values='bytes')[CODECS]
lo=df.pivot_table(index=['model','dataset'],columns='codec',values='lo')[CODECS]
ret=df.pivot_table(index=['model','dataset'],columns='codec',values='ret')[CODECS]
print("compression x float32 (median):", (df.groupby('codec').comp.median()).round(2).to_dict())
order=['binary','lloyd2(7)','int4','int4(mean)','int8(per_vector)','float16']  # cheapest first (int4 before int4(mean) by 1 byte)
def pick(tau,rule):
    out=[]
    for idx in ret.index:
        ok=None
        for c in order:
            val=ret.loc[idx,c] if rule=='point' else lo.loc[idx,c]
            if val>=tau: ok=c;break
        out.append(ok or 'none(float32)')
    return pd.Series(out,index=ret.index)
res={}
for rule,taus in [('point',[0.995,0.99,0.98,0.95]),('lowerCI',[0.99,0.98,0.95])]:
    for t in taus:
        s=pick(t,rule); res[(rule,t)]=s
        vc=s.value_counts(); print(rule,t,vc.to_dict())
sel=res[('point',0.99)]
# per-dataset breakdown
x=pd.DataFrame({'best':sel}).reset_index(); x['ds']=x.dataset.str.extract(r'_(docvqa|infovqa|scifact)')[0]
print(pd.crosstab(x.ds,x.best))
# mean compression at the selected codec (bytes-vs-float32) 
comp=df.pivot_table(index=['model','dataset'],columns='codec',values='comp')
for (rule,t),s in res.items():
    c=[comp.loc[i,v] if v in comp.columns else 1 for i,v in s.items()]
    print(rule,t,'median compression x',np.median(c).round(1),'mean',np.mean(c).round(2),'geo',np.exp(np.mean(np.log(c))).round(2))
# agreement of choice across datasets within model (point .99), image models
x['model']=x.model
im=x[x.ds!='scifact'].pivot_table(index='model',columns='ds',values='best',aggfunc='first').dropna()
print("image models with both:",len(im),"same choice:",(im.docvqa==im.infovqa).mean().round(3))
print(pd.crosstab(im.docvqa,im.infovqa))
# cheapest-rank agreement (ordinal): codec rank by bytes
rank={c:i for i,c in enumerate(order+['none(float32)'])}
rd=im.docvqa.map(rank); ri=im.infovqa.map(rank)
print("Spearman of choice rank docvqa vs infovqa:",stats.spearmanr(rd,ri))
# reproducibility of binary loss and lloyd2 loss across datasets (image models)
for c in ['binary','lloyd2(7)','int4(mean)','int4']:
    p=w[w.ds!='scifact'].pivot_table(index='model',columns='ds',values=c).dropna()
    r=stats.spearmanr(p.docvqa,p.infovqa); 
    bs=[]; 
    for _ in range(2000):
        i=rng.integers(0,len(p),len(p)); bs.append(stats.spearmanr(p.docvqa.values[i],p.infovqa.values[i])[0])
    print(c,'cross-dataset Spearman of retention n=%d rho=%.3f boot=%s'%(len(p),r[0],np.nanpercentile(bs,[2.5,97.5]).round(2)))
# sibling-pair: exact weights duplicates (colbert-ir vs lightonai colbertv2.0, tsystems merged, colpali hf vs orig)
pairs=[('vidore/colpali-v1.2','vidore/colpali-v1.2-hf'),('vidore/colpali-v1.3','vidore/colpali-v1.3-hf'),('vidore/colqwen2-v0.1','vidore/colqwen2-v1.0-hf'),('tsystems/colqwen2.5-3b-multilingual-v1.0','tsystems/colqwen2.5-3b-multilingual-v1.0-merged'),('colbert-ir/colbertv2.0','lightonai/colbertv2.0')]
for a,bb in pairs:
    for ds in ['docvqa','infovqa','scifact']:
        A=w[(w.model==a)&(w.ds==ds)]; B=w[(w.model==bb)&(w.ds==ds)]
        if len(A) and len(B): print(a.split('/')[1],'vs',bb.split('/')[1],ds,'binary',A.binary.round(4).values,B.binary.round(4).values,'lloyd2',A['lloyd2(7)'].round(4).values,B['lloyd2(7)'].round(4).values)
