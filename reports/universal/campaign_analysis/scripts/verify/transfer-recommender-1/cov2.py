import json, collections, re
m = json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
from optivision.adapters import MODELS
k = 'nvidia/nemotron-colembed-vl-8b-v2'
ds = list(m[k])[0]
print([ (r['label']) for r in m[k][ds]['rows']])
print(m[k][ds]['rows'][30]['pipeline'])
k = 'vidore/colpali'; ds = list(m[k])[0]
print([ (r['label']) for r in m[k][ds]['rows']])
k='tomoro'
for k in m:
    if 'tomoro-colqwen3-embed-4b' in k: print([r['label'] for r in m[k][list(m[k])[0]]['rows']])
