from common import *
M=measured(); O=outcomes()
print(len(M), sum(len(r['rows']) for m in M.values() for r in m.values()))
for mid,d in sorted(M.items()):
    for ds,v in d.items():
        rs=v['rows']
        ret=[r['retention'] for r in rs]
        print(f"{mid:55s}{ds:10s} base={v['baseline']:.4f} nd={v['n_docs']} nq={v['n_queries']} rows={len(rs)} maxret={max(ret):.3f} minret={min(ret):.3f} in_camp={mid in O}")
print([k for k in O if k not in M])
d=list(M.values())[0]; print(json.dumps(list(d.values())[0]['rows'][0],indent=1)); print(list(d.values())[0]['note'])
