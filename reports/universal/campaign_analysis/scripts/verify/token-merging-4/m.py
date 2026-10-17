import json
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
for m,v in d.items():
    if len(v)<2: continue
    for ds in v:
        L=[r['label'] for r in v[ds]['rows']]
        miss=[l for l in ['hierarchical_merge(0.5)','hierarchical_merge(0.33)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','adaptive_merge(0.6)'] if l not in L]
        if miss: print(m,ds[-20:],miss,[l for l in L if 'merge' in l and '>' not in l][:6], v[ds]['n_docs'])
