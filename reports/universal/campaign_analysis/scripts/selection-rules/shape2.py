import json,sys
d=json.load(open(sys.argv[1]))
for k,v in d.items():
    if k!='results' and k!='candidates': print(k, json.dumps(v)[:600])
r=d['results']; print(type(r), list(r)[:5] if isinstance(r,dict) else len(r))
def walk(o,dep=0):
    if isinstance(o,dict):
        for i,(k,v) in enumerate(o.items()):
            if i<2: print(' '*dep,k,type(v).__name__,len(v) if hasattr(v,'__len__') else v); walk(v,dep+2)
    elif isinstance(o,list) and o: print(' '*dep,'item0',json.dumps(o[0])[:300])
walk(r)
c=d['candidates']; print(type(c),len(c), json.dumps(c[0] if isinstance(c,list) else list(c.items())[0])[:500])
