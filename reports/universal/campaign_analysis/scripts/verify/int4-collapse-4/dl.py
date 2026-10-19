import json
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
H=json.load(open("C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json"))
h={x["id"]:x["downloads"] for x in H}
miss=[m for m in M if m not in h];print("missing from hub",miss)
tot=sum(h.get(m,0) for m in M);print(tot)
def S(ids):return sum(h.get(m,0) for m in ids)
col=["nvidia/nemotron-colembed-vl-4b-v2","nvidia/nemotron-colembed-vl-8b-v2","mixedbread-ai/mxbai-edge-colbert-v0-17m","lightonai/LateOn"]
print("collapsed",S(col)/tot)
for th in (.95,.99):
    d=[m for m in M for dd in [M[m]] if (lambda r:r and sum(r)/len(r)<th)([x["retention"] for v in dd.values() for x in v["rows"] if x["label"]=="int4"])]
    print(th,len(d),S(d)/tot, sorted(d))
print(sorted(h.items(),key=lambda x:-x[1])[:5])
