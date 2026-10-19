import json
from collections import Counter
d=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
units={(m,ds):{r['label']:r for r in v['rows']} for m,x in d.items() for ds,v in x.items()}
print(len(units), Counter(ds for m,ds in units))
c=Counter(l for u in units.values() for l in u)
print(len(c)); print(sorted(c.items(),key=lambda t:-t[1])[:80])
