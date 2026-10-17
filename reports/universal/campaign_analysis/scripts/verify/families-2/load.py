import json,collections
m=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
img={k:v for k,v in m.items() if 'vidore/docvqa_test_subsampled' in v}
cnt=collections.Counter()
for k,v in img.items():
    for d in v.values():
        for l in {r['label'] for r in d['rows']}: cnt[l]+=1
n=2*len(img)
for l,c in sorted(cnt.items(),key=lambda x:-x[1]): print(c,n,l)
