import json,glob
for f in glob.glob('geometry/*.json')[:7]:
    d=json.load(open(f));print(f,list(d.keys())[:20])
d=json.load(open('geometry/text-answerai-colbert-small-v1-scifact.json'))
print(json.dumps(d,indent=0)[:3000])
