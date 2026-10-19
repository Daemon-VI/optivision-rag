import json,glob,collections
f=glob.glob('reports/universal/followup/f1*/*/selection_rules/*.json')[0]
d=json.load(open(f))
print(d.keys(), len(d['results']))
print(collections.Counter((r['reference'],r['target']) for r in d['results']))
r=d['results'][0]; print({k:v for k,v in r.items() if k!='outcomes'})
print(d['candidates'][:3])
w=json.load(open(glob.glob('reports/universal/followup/f1*/*/wide/*.json')[0]))
print([(x['label'],x['group'],x['compression_vs_float32']) for x in w['rows']])
print(w['tiered'] if 'tiered' in w else '')
