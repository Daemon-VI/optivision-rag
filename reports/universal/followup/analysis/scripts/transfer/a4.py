import pandas as pd,numpy as np,json,glob
from scipy.stats import spearmanr
df=pd.read_pickle('df.pkl'); df=df[df.label!='float32']
df['sS']=df.slo>=0.95; df['sN']=df.nlo>=0.95
for thr in [0.95,0.97,0.98,0.99]:
    s=df[df.slo>=thr]; print('slo>=',thr,'n',len(s),'N lo>=.95',(s.nlo>=.95).mean().round(3),'false',(s.nlo<.95).sum(),'worst point n',s.n.min().round(3),'n<0.90',(s.n<.90).sum(), 'median drop',(s.n-s.s).median().round(4))
s=df[df.sS]; print('safeS: nlo<0.90',(s.nlo<.90).sum(),' point n<0.95',(s.n<.95).sum())
# safe-S excluding anything containing hierarchical_merge(0.33)/adaptive/ project : by family
def fam(l):
    if l in('float16','int8(per_vector)'):return 'f16/int8'
    if l=='int4(mean)':return 'int4(mean)'
    if '>' in l: return 'chain'
    return 'other'
# model-level rank corr of same pipeline across models between datasets
for l in ['int4','binary','lloyd2(7)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','int4(mean)']:
    x=df[df.label==l]; print(l,'spearman across 8 models SciFact vs NFC',round(spearmanr(x.s,x.n)[0],2),' pearson',round(np.corrcoef(x.s,x.n)[0,1],2))
# Spearman within model restricted to pipelines with s in 0.5..1 (non-degenerate)
r=[]
for m,x in df.groupby('m'):
    y=x[(x.comp>1.5)&(x.comp<40)]; r.append(round(spearmanr(y.s,y.n)[0],2))
print('within-model rho, comp 1.5-40:',r)
# mean abs delta, by family at comp<=8
print(df[df.comp<=8].groupby('label').d.agg(['count','mean']).round(3).T.to_string())
# qrels
for p in ['../../../../campaign/colbert-ir--colbertv2.0','../../../../followup/f1_text_nfcorpus_a/colbert-ir--colbertv2.0']:
    for q in glob.glob(p+'/*qrels*'):
        d=json.load(open(q)); 
        v=list(d.values()) if isinstance(d,dict) else d
        print(q.split('/')[-1],type(d),len(d), str(v[0])[:100])
