import json,sys
d=json.load(open(sys.argv[1]))
def sh(o,dep=0,k='',maxd=7):
    p=' '*dep+str(k)
    if isinstance(o,dict):
        print(p,'{',len(o),list(o)[:6] if dep>=0 else '')
        if dep<maxd*2:
            for i,(kk,v) in enumerate(o.items()):
                if i<3: sh(v,dep+2,kk)
    elif isinstance(o,list):
        print(p,'[',len(o))
        if o: sh(o[0],dep+2,'0')
    else: print(p,repr(o)[:100])
sh(d)
