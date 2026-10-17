from common import *
pd.set_option('display.width',300,'display.max_rows',500,'display.max_columns',40)
mp=df[df.names=='hierarchical_merge>project']
print(mp.groupby(['D','method','k']).agg(n=('ret','size'),mean=('ret','mean'),models=('model','nunique')).to_string())
mo=df[(df.names=='hierarchical_merge')]
print(mo.label.value_counts())
print(mp.label.value_counts().head(5))
print(df[df.names.str.startswith('hierarchical_merge>project')].names.value_counts())
