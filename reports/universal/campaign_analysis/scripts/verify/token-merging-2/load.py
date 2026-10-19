import json,collections,numpy as np,pandas as pd
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rows=[]
for m,ds in d.items():
    for k,v in ds.items():
        base=[r for r in v["rows"] if not r["pipeline"]["stages"]][0]
        for r in v["rows"]:
            st=r["pipeline"]["stages"]
            if len(st)==1 and st[0]["stage"]=="hierarchical_merge":
                rows.append(dict(model=m,ds=k.split("/")[1].split("_")[0],ratio=st[0]["ratio"],label=r["label"],ret=r["retention"],vpd=r["vectors_per_doc"],bv=base["vectors_per_doc"],note=v.get("note","")[:12]))
df=pd.DataFrame(rows); df["loss"]=1-df.ret; df["keep"]=df.vpd/df.bv
df.to_pickle("h.pkl")
print(df.groupby("ds").agg(n=("loss","size"),m=("model","nunique")))
print(df.ratio.value_counts()); print(df.label.value_counts().head(10))
print(df.groupby("ds").keep.describe())
