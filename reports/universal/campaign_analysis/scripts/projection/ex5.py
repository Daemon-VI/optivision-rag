from common import *
pd.set_option('display.width',250,'display.max_rows',500)
print(dim.isna().sum(), dim.value_counts().sort_index().to_dict())
print(bs.evidence.head(3).tolist())
print(pp.groupby(['method','k']).size().unstack(0))
print(pp.groupby('model').apply(lambda g: sorted(set(zip(g.k,g.method)))).head(70).to_string())
