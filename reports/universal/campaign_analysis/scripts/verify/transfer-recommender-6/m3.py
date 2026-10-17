import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
for m in ['vidore/colpali-v1.3-merged','vidore/colpali']:
    x=d[m]['vidore/docvqa_test_subsampled']['rows']
    for r in x[:6]+x[-3:]: print(r['label'],json.dumps(r['pipeline']))
