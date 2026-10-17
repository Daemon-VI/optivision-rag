import json,sys,collections
d=json.load(open(sys.argv[1]))
r=d['results']
print(collections.Counter((x['reference'],x['rule'],x['target']) for x in r).most_common(3))
print(sorted(set(x['reference'] for x in r)),sorted(set(x['target'] for x in r)),sorted(set(x['rule'] for x in r)))
print(json.dumps(r[0])[:1500])
print([ (x['rule'],x['target'],x['selected'],x['met']) for x in r if x['reference']=='labels' and x['target']==0.95])
