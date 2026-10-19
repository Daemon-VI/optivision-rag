import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
for m,v in d.items():
    for ds,x in v.items():
        ls=[r['label'] for r in x['rows']]
        if 'float32' not in ls: print(m,ds,ls[:4],x['baseline'],x['rows'][0]['compression'],x['rows'][0]['vectors_per_doc'])
