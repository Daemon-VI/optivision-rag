import json, collections
m = json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
from optivision.adapters import MODELS
c = collections.defaultdict(collections.Counter); n = collections.Counter()
for k, v in m.items():
    for ds, d in v.items():
        mo = MODELS[k].modality; n[mo] += 1
        for r in d['rows']: c[mo][r['label']] += 1
print(n)
for mo in c:
    print(mo, len(c[mo]), sum(1 for l in c[mo] if c[mo][l] == n[mo]))
a = set(l for l in c['image'] if c['image'][l] >= 70); b = set(l for l in c['text'] if c['text'][l] >= 12)
print(len(a), len(b), len(a & b))
print(sorted(a)[:80]); print(sorted(b))
