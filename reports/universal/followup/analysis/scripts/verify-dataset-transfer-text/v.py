import json,glob,os,numpy as np,pandas as pd
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
K='retention:ndcg@10';C='retention_ci:ndcg@10'
rows=[]
for p in glob.glob(R+'followup/f[12]_*/*/wide/*nfcorpus.json'):
    m=os.path.basename(os.path.dirname(os.path.dirname(p)))
    c=glob.glob(R+'campaign/'+m+'/wide/*scifact.json')
    if m.startswith('answerdotai'): c=glob.glob(R+'followup/f3b*/'+m+'/wide/*scifact.json')
    a=json.load(open(c[0]));b=json.load(open(p))
    print(m.split('--')[1],a['n_docs'],a['n_queries'],b['n_docs'],b['n_queries'])
    A={r['label']:r for r in a['rows']};B={r['label']:r for r in b['rows']}
    dupA=len(a['rows'])-len(A);dupB=len(b['rows'])-len(B)
    if dupA or dupB:print('dup labels',dupA,dupB)
    for l in A:
        if l in B and l!='float32':
            x,y=A[l],B[l]
            rows.append(dict(m=m.split('--')[1],label=l,comp=x['compression_vs_float32'],compN=y['compression_vs_float32'],s=x[K],slo=x[C][0],shi=x[C][1],n=y[K],nlo=y[C][0],nhi=y[C][1]))
df=pd.DataFrame(rows);df['d']=df.n-df.s
print(len(df),(abs(df.comp-df.compN)>1e-6*df.comp).sum())
df['sS']=df.slo>=.95;df['sN']=df.nlo>=.95
for t in [.95,.96,.97,.98,.99]:
    s=df[df.slo>=t];print(t,len(s),(~s.sN).sum(),'pointN<.95',(s.n<.95).sum())
fs=df[df.sS&~df.sN];print(fs.groupby('m').size().to_dict(),df[df.sS].groupby('m').size().to_dict())
print(fs.sort_values('n').head(6)[['m','label','s','n','nlo']].round(3))
print('nlo<.9',(fs.nlo<.9).sum(),'n<.95',(fs.n<.95).sum())
# excluding answerai
x=df[df.m!='answerdotai--answerai-colbert-small-v1'] if False else df[~df.m.str.contains('answerai')]
s=x[x.slo>=.95];print('excl answerai',len(s),(~s.sN).sum(),'at .97',(x.slo>=.97).sum(),((x.slo>=.97)&~x.sN).sum())
# 0.97 failures
f97=df[(df.slo>=.97)&~df.sN];print(f97[['m','label','s','slo','n','nlo']].round(3))
# asymmetry: 27 reverse and a null baseline - threshold dependence of N: regression to mean? false-safe w/ N retention point
core=['float16','int8(per_vector)','int4(mean)','int4','binary','lloyd2(7)','hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','project(64)']
for l in core:
    z=df[df.label==l];dis=((z.nhi<z.slo)|(z.nlo>z.shi)).sum();print(l,len(z),round(z.s.mean(),3),round(z.n.mean(),3),round(z.d.min(),3),round(z.d.max(),3),'dis',dis)
print(df[df.label=='int4'][['m','s','n']].round(3))
# comp bins
df['bin']=pd.cut(df.comp,[0,2.5,4.5,8,16.5,40,100,1e4]);print(df.groupby('bin',observed=True).d.agg(['count','mean','median']).round(3))
from scipy.stats import spearmanr
for m,g in df.groupby('m'):print(m[:20],round(spearmanr(g.s,g.n)[0],2),round(spearmanr(g[(g.comp>1.5)&(g.comp<40)].s,g[(g.comp>1.5)&(g.comp<40)].n)[0],2))
print('>=40 safe',((df.comp>=40)&(df.sS)).sum(),((df.comp>=40)&df.sN).sum())
# mean 'd' vs comp excluding binary/merge-labelled
df['kind']=np.where(df.label.str.contains('binary|merge'),'bin/merge','other')
print(df.groupby(['kind','bin'],observed=True).d.agg(['count','mean']).round(3))
print('----')
pd.set_option('display.width',250)
a=df[df.m.str.contains('answerai')&df.label.str.contains('int4\(mean\)|merge\(0.33\)')];print(a[['label','s','slo','n','nlo']].round(3).to_string())
print(fs.sort_values('n').head(6)[['m','label','s','n','nlo']].round(3).to_string())
print(df[df.label.str.contains('merge')&df.label.str.contains('int4\(mean\)')].groupby('m').apply(lambda g:pd.Series(dict(k=len(g),mind=g.d.min()))).round(3))
