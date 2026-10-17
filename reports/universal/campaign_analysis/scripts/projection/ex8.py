from common import *
pd.set_option('display.width',300,'display.max_rows',500,'display.max_columns',40)
# merge ratio keyed
def key(l): return re.match(r'(hierarchical_merge\([\d.]+\))',l).group(1) if l.startswith('hier') else None
d2=df[df.names.isin(['hierarchical_merge','hierarchical_merge>project'])].copy()
d2['mk']=d2.label.map(key)
mo=d2[d2.names=='hierarchical_merge'].set_index(['model','ds','mk']).ret
mp=d2[d2.names=='hierarchical_merge>project'].copy()
mp['mret']=[mo.get((a,b,c),np.nan) for a,b,c in zip(mp.model,mp.ds,mp.mk)]
mp['rel']=mp.ret/mp.mret
print(mp.mret.isna().sum())
mp['frac']=mp.k/mp.D
print(mp[(mp.k>=mp.D)].groupby(['D','k']).rel.describe())
print(df[(df.method=='random')&(df.names=='project')&(df.ret<0.1)][['model','ds','D','k','ret','lo','hi']])
print(pp[(pp.model.str.contains('mxbai-edge'))][['model','ds','D','k','method','ret']])
print(df[df.model.str.contains('mxbai-edge-colbert-v0-17m')][['label','ret']].head(40))
