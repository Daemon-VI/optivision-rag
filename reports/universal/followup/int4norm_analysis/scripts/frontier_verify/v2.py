import json,glob
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/f9b_int4norm_text/'
for p in sorted(glob.glob(R+'*/int4norm/*.json')):
    d=json.load(open(p));g={r['label']:r['retention:ndcg@5'] for r in d['rows']}
    a=g['hierarchical_merge(0.25) > int8(per_vector)']
    if a<.8 or 'LFM2' in p or 'answerai' in p or 'mxbai-edge' in p: print(p.split('/')[-1][:40],round(a,3),round(g['hierarchical_merge(0.25) > int4norm(mean)'],3),round(g['hierarchical_merge(0.33) > int8(per_vector)'],3),round(g['lloyd2(7)'],3))
