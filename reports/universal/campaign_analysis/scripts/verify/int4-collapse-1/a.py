import json,re
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
print(len(M))
for m in ["nvidia/nemotron-colembed-vl-4b-v2","nvidia/nemotron-colembed-vl-8b-v2"]:
    for d,v in M[m].items():
        print(m,d,v['baseline'])
        for r in v['rows']:
            if len(r['pipeline']['stages'])<=2 and r['retention']<2: print('  ',r['label'],round(r['retention'],3),r['retention_interval'])
