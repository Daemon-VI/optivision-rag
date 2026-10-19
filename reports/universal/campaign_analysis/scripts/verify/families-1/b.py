import json
d=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
rows=[]
for m,ds in d.items():
    r4={};rm={};base=set()
    for k,v in ds.items():
        for r in v["rows"]:
            if r["label"]=="int4": r4[k.split("/")[-1][:8]]=(r["retention"],r["retention_interval"])
            if r["label"]=="int4(mean)": rm[k.split("/")[-1][:8]]=r["retention"]
        base.add(v.get("note","")[:25])
    if not r4: print("NO int4:",m,list(ds)); continue
    rows.append((min(x[0] for x in r4.values()),m,r4,rm,base))
rows.sort()
for mn,m,r4,rm,b in rows[:14]: print(round(mn,3),m,{k:(v[0],v[1]) for k,v in r4.items()},rm)
print(len(rows),[ (round(x[0],3),x[1]) for x in rows[10:16]])
print("int4mean range",min(min(x[3].values()) for x in rows),max(max(x[3].values()) for x in rows))
print("n<0.9",sum(x[0]<0.9 for x in rows), "n<0.98",sum(x[0]<0.98 for x in rows))
print("\n".join(sorted(x[1] for x in rows)))
