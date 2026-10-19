import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
df=load()
old=df[df.label.str.contains('ward|int8/vec|adaptive r')]
print(old.groupby(['model','ds']).size().to_string()[:1500])
m=old.model.iloc[0]
print(sorted(df[(df.model==m)].label.unique()))
print(df[df.label.str.contains('project')].label.value_counts().to_string())
