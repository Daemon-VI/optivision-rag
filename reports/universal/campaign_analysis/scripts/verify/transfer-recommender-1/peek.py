import json
from optivision.adapters import MODELS
m=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
print(len(m))
k=next(iter(m)); print(k, list(m[k]))
d=m[k][list(m[k])[0]]; print({a:b for a,b in d.items() if a!='rows'}); print(d['rows'][:2]); print(len(d['rows']))
print(type(MODELS))
x=MODELS[k] if isinstance(MODELS,dict) else MODELS[0]; print(x)
