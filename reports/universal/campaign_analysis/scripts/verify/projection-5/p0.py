import json
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
k='ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1'
for ds,v in m[k].items():
    print(ds,v['baseline'],[ (r['label'],r['pipeline']) for r in v['rows'] if 'proj' in r['label']][:12])
    print(len(v['rows']), [r['label'] for r in v['rows']][:40])
