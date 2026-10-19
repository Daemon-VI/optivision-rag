from load import *
df,d=load()
print(df.iloc[0].base)
x=df[(df.stage=='hierarchical_merge')&df.ratio.isna()];print(x.label.value_counts().head(),x.model.nunique())
r=df[df.ratio.notna()|df.radius.notna()]
for rt in [0.1,0.25,0.5]:
    print(rt, set(df.model)-set(df[(df.ratio==rt)].model))
print(df.groupby('ds').n_docs.unique())
print(df[df.stage=='adaptive_merge'].groupby(['ds','radius']).size())
# duplicates
print(df.duplicated(['model','ds','label']).sum())
print(df[df.label.str.contains('hier')&df.ratio.isna()].label.unique())
