import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
df=load()
print(df.shape, df.model.nunique())
print(df.groupby('ds').model.nunique())
print(df.note.value_counts().head(8))
# index
idx={(r.model,r.ds,r.label):r for r in df.itertuples()}
combos=df[df.stages.map(len)>1]
print(combos.stages.map(len).value_counts())
# check missing singles
miss=0;tot=0
for r in combos.itertuples():
    for s in r.stages:
        tot+=1
        if (r.model,r.ds,s) not in idx: miss+=1
print('missing singles',miss,tot)
print(df.ret.describe()); print((df.ret>1).mean())
print(df[df.label.isin(['binary'])].ret.describe())
