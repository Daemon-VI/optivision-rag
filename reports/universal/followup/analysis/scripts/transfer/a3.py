import pandas as pd,numpy as np
df=pd.read_pickle('df.pkl'); df=df[df.label!='float32']
df['sS']=df.slo>=0.95; df['sN']=df.nlo>=0.95
print('pairs',len(df),'safe S',df.sS.sum(),'safe N',df.sN.sum())
fs=df[df.sS&~df.sN]; print('false-safe',len(fs),'; safe both',(df.sS&df.sN).sum(),'; N safe not S',(~df.sS&df.sN).sum())
print('false-safe: miss (point N), nlo, drop vs s');print(fs[['m','label','comp','s','slo','n','nlo']].round(3).sort_values('nlo').to_string())
print('false-safe by model',fs.groupby('m').size().to_dict(),' safeS by model',df[df.sS].groupby('m').size().to_dict())
print('false-safe by label count',fs.label.value_counts().head(15).to_dict())
# point-based: n<0.95 & s>=.95
fp=df[(df.s>=0.95)&(df.n<0.95)]; print('point-safe S false on N',len(fp),'of',(df.s>=0.95).sum(),'min n',fp.n.min())
# among safe-S, distribution of n point
ss=df[df.sS]; print('safe-S n point: min/median/mean',ss.n.min(),ss.n.median(),ss.n.mean(),'  frac nlo>=.95',ss.sN.mean())
# compression bins
df['bin']=pd.cut(df.comp,[0,2.5,4.5,8,16.5,40,100,1e4])
print(df.groupby('bin',observed=True).apply(lambda x:pd.Series(dict(n=len(x),safeS=x.sS.sum(),falsesafe=(x.sS&~x.sN).sum(),meand=x.d.mean(),meds=x.s.median(),medn=x.n.median()))).to_string())
# CI-overlap: how many differences are beyond CI widths
df['sig']=(df.nhi<df.slo)|(df.nlo>df.shi)
print('CI disjoint',df.sig.sum(),'of',len(df),'; N lower',((df.nhi<df.slo)).sum(),'N higher',(df.nlo>df.shi).sum())
core=['float16','int8(per_vector)','int4(mean)','int4','binary','lloyd2(7)','hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)','project(64)']
c=df[df.label.isin(core)]
t=c.groupby('label').agg(k=('m','count'),s=('s','mean'),n=('n','mean'),d=('d','mean'),sd_d=('d','std'),absd=('d',lambda x:x.abs().mean()),safeS=('sS','sum'),fs=('sS',lambda x:0),dis=('sig','sum'))
t['fs']=c[c.sS&~c.sN].groupby('label').size(); print(t.fillna(0).round(3).to_string())
print('int4 per model');print(df[df.label=='int4'][['m','s','slo','n','nlo']].round(3).to_string())
print('int4(mean) per model');print(df[df.label=='int4(mean)'][['m','s','slo','n','nlo']].round(3).to_string())
print('binary');print(df[df.label=='binary'][['m','s','slo','n','nlo']].round(3).to_string())
