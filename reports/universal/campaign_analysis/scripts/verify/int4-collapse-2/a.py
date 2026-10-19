import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
m=next(iter(d)); ds=next(iter(d[m]))
print(m,ds,[ (r['label']) for r in d[m][ds]['rows']][:60])
