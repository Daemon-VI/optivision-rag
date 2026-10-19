from common import *
df=long_df()
P=mat(df)
pd.set_option('display.width',250)
t=P[['int4','int4(mean)','binary','lloyd2(7)','hierarchical_merge(0.1)','hierarchical_merge(0.33)']].round(3)
t['grp']=[info(m)['group'] for m,_ in t.index]
t['f16']=[info(m)['f16'] for m,_ in t.index]
t['dim']=[info(m)['dim'] for m,_ in t.index]
print(t.sort_values(['grp','int4']).to_string())
