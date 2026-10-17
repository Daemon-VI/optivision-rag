from common import *
df=long_df()
P=mat(df)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
cols=['project(128)','project(80)','project(64)','project(32)','project(32, random)','hierarchical_merge(0.33) > project(128)','hierarchical_merge(0.33) > project(32)','hierarchical_merge(0.33) > project(16)']
t=P[[c for c in cols if c in P.columns]].copy()
t.columns=['p128','p80','p64','p32','p32r','hm33p128','hm33p32','hm33p16'][:t.shape[1]] if t.shape[1]==8 else t.columns
t['dim']=[info(m)['dim'] for m,_ in t.index]; t['grp']=[GROUP2.get(m,'?') for m,_ in t.index]
t['f16']=[info(m)['f16'] for m,_ in t.index]
print(t.round(3).sort_values(['dim','grp']).to_string())
