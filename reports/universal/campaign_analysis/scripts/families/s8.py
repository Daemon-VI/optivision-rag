from common import *
import json
df=long_df(); L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
S=loss.xs('sci',level='ds'); SE=se_mat(df,L).xs('sci',level='ds').loc[S.index]
ms=list(S.index); n=len(ms)
X=S.values; D=np.sqrt(((X[:,None]-X[None])**2).mean(-1))
iu=np.triu_indices(n,1); d=D[iu]
a=ms.index('colbert-ir/colbertv2.0'); b=ms.index('lightonai/colbertv2.0')
print('text models',n,'pairs',len(d),'colbertv2 pair RMSD %.4f ; rank %d of %d (1=closest); median pair %.4f, min %.4f'%(D[a,b],(d<D[a,b]).sum()+1,len(d),np.median(d),d.min()))
# per-label diffs and CI overlap
raw=df[(df.ds=='sci')&df.model.isin(['colbert-ir/colbertv2.0','lightonai/colbertv2.0'])]
A=raw[raw.model=='colbert-ir/colbertv2.0'].set_index('label'); B=raw[raw.model=='lightonai/colbertv2.0'].set_index('label')
labs=[l for l in A.index if l in B.index]
disjoint=[l for l in labs if A.loc[l,'hi']<B.loc[l,'lo'] or B.loc[l,'hi']<A.loc[l,'lo']]
print('labels shared',len(labs),'with non-overlapping retention intervals',len(disjoint),disjoint)
print('mean ret colbert-ir %.4f lightonai %.4f ; max abs diff %.4f'%(A.loc[labs,'ret'].mean(),B.loc[labs,'ret'].mean(),(A.loc[labs,'ret']-B.loc[labs,'ret']).abs().max()))
print('vpd',A.vpd.iloc[0],B.vpd.iloc[0],'baseline',A.base.iloc[0],B.base.iloc[0])
dd=(A.loc[labs,'ret']-B.loc[labs,'ret']); print(dd.sort_values().round(3).head(5).to_dict(),dd.sort_values().round(3).tail(3).to_dict())
# raw wide files
import glob
for f in glob.glob(PKG+'/reports/universal/campaign/*colbertv2.0/wide/*.json'):
    j=json.load(open(f)); print(f.replace(chr(92),'/').split('campaign/')[1],j['baseline'] if 'baseline' in j else None)
for f in glob.glob(PKG+'/reports/universal/campaign/*colbertv2.0/run.json'):
    j=json.load(open(f)); print(f.replace(chr(92),'/').split('campaign/')[1], {k:j[k] for k in j if k in('spec','check_retriever')})
# does baseline gap or vpd gap predict profile distance among text pairs? Spearman of RMSD with |dvpd|, |dbase|
from scipy.stats import spearmanr
base=np.array([df[(df.model==m)&(df.ds=='sci')].base.iloc[0] for m in ms]); vp=np.array([df[(df.model==m)&(df.ds=='sci')].vpd.iloc[0] for m in ms])
db=np.abs(base[:,None]-base[None])[iu]; dv=np.abs(np.log(vp)[:,None]-np.log(vp)[None])[iu]
print('text pairs: spearman RMSD vs |dbaseline| %.3f (p %.3f), vs |dlog vpd| %.3f (p %.3f)'%(*spearmanr(d,db),*spearmanr(d,dv)))
# restricted: drop collapse-driven labels? use only merge labels
Lm=[l for l in L if 'merge' in l and '>' not in l]
Xm=S[Lm].values; Dm=np.sqrt(((Xm[:,None]-Xm[None])**2).mean(-1)); dm=Dm[iu]
print('merge-only labels',Lm,': colbertv2 pair RMSD %.4f rank %d/%d'%(Dm[a,b],(dm<Dm[a,b]).sum()+1,len(dm)))
# lineage pairs ranks
for nm,(x,y) in {'LFM2 vs LFM2.5':('LiquidAI/LFM2-ColBERT-350M','LiquidAI/LFM2.5-ColBERT-350M'),'mxbai 17m vs 32m':('mixedbread-ai/mxbai-edge-colbert-v0-17m','mixedbread-ai/mxbai-edge-colbert-v0-32m'),'LateOn vs mLateOn':('lightonai/LateOn','lightonai/mLateOn')}.items():
    i,j=ms.index(x),ms.index(y); print(nm,'RMSD %.4f rank %d/%d'%(D[i,j],(d<D[i,j]).sum()+1,len(d)))
