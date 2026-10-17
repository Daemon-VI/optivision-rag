from common import *
from scipy.stats import binomtest
rng=np.random.default_rng(10)
df=long_df(); L=L1b(df)
P=mat(df)[L].dropna(); loss=1-P
PAIRS=[('nemotron 4b>8b','nvidia/nemotron-colembed-vl-4b-v2','nvidia/nemotron-colembed-vl-8b-v2'),
('EVIE 4.5B>8B','tencent/EVIE-4.5B','tencent/EVIE-8B'),
('Vultron 0.8B>4.5B','vultr/VultronRetrieverFlash-Qwen3.5-0.8B','vultr/VultronRetrieverCore-Qwen3.5-4.5B'),
('Vultron 4.5B>8B','vultr/VultronRetrieverCore-Qwen3.5-4.5B','vultr/VultronRetrieverPrime-Qwen3.5-8B'),
('Sauerkraut 2b>4b','VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1','VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1'),
('tomoro 4b>8b','TomoroAI/tomoro-colqwen3-embed-4b','TomoroAI/tomoro-colqwen3-embed-8b'),
('nomic 3b>7b','nomic-ai/colnomic-embed-multimodal-3b','nomic-ai/colnomic-embed-multimodal-7b'),
('NeoMME 260M>800M','Hcompany/NeoMME-260M-Retriever-ST-late','Hcompany/NeoMME-800M-Retriever-ST-late'),
('mxbai-edge 17m>32m','mixedbread-ai/mxbai-edge-colbert-v0-17m','mixedbread-ai/mxbai-edge-colbert-v0-32m')]
sets={'all L1b':L,'merge-only':[l for l in L if '>' not in l and 'merge' in l],'codec-only':['binary','int4(mean)','int8(per_vector)','lloyd2(7)'],'binary':['binary']}
rows=[]
for nm,a,b in PAIRS:
    dss=['sci'] if 'mxbai' in nm else ['doc','info']
    for s,labs in sets.items():
        dv=[]; 
        for ds in dss:
            dv.append(loss.loc[(b,ds)][labs].mean()-loss.loc[(a,ds)][labs].mean())
        rows.append((nm,s,np.mean(dv)*100,*(x*100 for x in dv),a in [i[0] for i in loss.index]))
R=pd.DataFrame(rows,columns=['pair','set','dloss_pts_mean','d1','d2','_']).drop(columns='_')
piv=R.pivot(index='pair',columns='set',values='dloss_pts_mean').loc[[p[0] for p in PAIRS]]
pd.set_option('display.width',200); print('large minus small loss, points of retention (positive = larger model compresses WORSE)'); print(piv.round(2))
for s in sets:
    x=piv[s].values; k=(x>0).sum()
    bs=[rng.choice(x,len(x)).mean() for _ in range(5000)]
    print('%-12s mean %+.2f pts [%+.2f,%+.2f] bootstrap over pairs; larger worse in %d/%d (sign test p=%.3f)'%(s,x.mean(),*np.percentile(bs,[2.5,97.5]),k,len(x),binomtest(k,len(x),.5).pvalue))
# exclude text pair
x=piv.drop('mxbai-edge 17m>32m')
for s in sets: print('image-only',s,'mean %+.2f larger worse %d/%d'%(x[s].mean(),(x[s]>0).sum(),len(x)))
# Vultron monotonic
for ds in ['doc','info']:
    print('Vultron mean L1b loss pts',ds,[round(100*loss.loc[(m,ds)].mean(),2) for m in ['vultr/VultronRetrieverFlash-Qwen3.5-0.8B','vultr/VultronRetrieverCore-Qwen3.5-4.5B','vultr/VultronRetrieverPrime-Qwen3.5-8B']])
# across all models: size vs loss, image, using known sizes; partial for dataset
sz={m:size_b(m) for m in set(i[0] for i in loss.index) if size_b(m)}
rowsz=[]
for (m,ds) in loss.index:
    if m in sz and ds!='sci': rowsz.append((m,ds,np.log10(sz[m]),loss.loc[(m,ds)].mean()*100,GROUP2.get(m,'single'),int(info(m)['f16'])))
Z=pd.DataFrame(rowsz,columns=['m','ds','lsz','loss','grp','f16'])
from scipy.stats import spearmanr
print('image models with known size',Z.m.nunique(),'spearman(loss, log size) within doc: %.3f p %.3f ; info: %.3f p %.3f'%(*spearmanr(Z[Z.ds=='doc'].lsz,Z[Z.ds=='doc'].loss),*spearmanr(Z[Z.ds=='info'].lsz,Z[Z.ds=='info'].loss)))
