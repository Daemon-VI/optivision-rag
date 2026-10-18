import json,glob,os
rows=[]
for f in glob.glob('followup/f[1-3]*/*/wide/*.json')+glob.glob('campaign/*/wide/*.json'):
    d=json.load(open(f)); 
    if 'rows' not in d: continue
    R={r['label']:r for r in d['rows']}
    if 'int4' not in R: continue
    b=d['baseline']; parts=f.replace(chr(92),'/').split('/')
    g=f.replace('wide','geometry').replace('campaign','campaign')
    ge=None
    if os.path.exists(g):
        ge=json.load(open(g))
    def ret(l): return R[l]['retention:ndcg@10'] if l in R else float('nan')
    rows.append((parts[-3] if 'followup' in f else parts[-3], parts[-1][:-5], d['dim'] if 'dim' in d else b['dim'], round(b['vectors_per_doc']), d['n_docs'], d['n_queries'], round(b['ndcg@10'],3), round(b['recall@1'],3), round(ret('int4'),3), round(ret('int4(mean)'),3), round(ret('lloyd2(7)'),3), round(ret('int8(per_vector)'),3), (round(ge['codecs']['int4']['score_pearson']['mean'],3) if ge else None)))
rows.sort(key=lambda r:(r[1].split('_')[-1],r[8]))
print("batch tag dim vpd ndocs nq nd10 r1 | int4 int4mean lloyd int8 pearson")
for r in rows: print(*r)
