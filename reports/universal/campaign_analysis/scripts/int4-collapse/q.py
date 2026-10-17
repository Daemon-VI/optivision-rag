import pandas as pd
m=pd.read_pickle('master.pkl')
for mod in ['vidore/colqwen2-v1.0-merged','vidore/colqwen2.5-v0.2','answerdotai/answerai-colbert-small-v1','vidore/colpali-v1.3-merged']:
    r=m[m.model==mod]
    for _,x in r.iterrows():
        print(mod,x.dataset,x.note, {k:round(x[k],3) for k in m.columns if (k.startswith('none') or k in('float16','int4','binary')) and not k.endswith(('_lo','_hi')) and x[k]==x[k]})
