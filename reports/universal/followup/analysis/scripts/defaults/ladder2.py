import json,collections,numpy as np
R='reports/universal/'
meta=json.load(open(R+'followup/analysis/scripts/defaults/units.json'))
K='retention_ci:ndcg@5'; P='retention:ndcg@5'
rungs=['float16','int8(per_vector)','int4(mean)','int4','lloyd2(7)','binary','hierarchical_merge(0.5)','hierarchical_merge(0.33) > int8(per_vector)','hierarchical_merge(0.25) > int8(per_vector)','hierarchical_merge(0.33) > int4(mean)','hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25)','hierarchical_merge(0.1)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.33) > binary','hierarchical_merge(0.25) > lloyd2(7)']
groups=collections.defaultdict(list)
for f,org,img,fol,dim in meta:
    d=json.load(open(f)); rows={r['label']:r for r in d['rows']}
    g='text' if not img else ('image128' if dim<=128 else 'image wide(>128)')
    groups[g].append((f,org,rows))
    groups['ALL'].append((f,org,rows))
out={}
for g,us in groups.items():
    print('==',g,len(us),'units, models',len({o for _,o,_ in us}))
    for r in rungs:
        lo=np.array([u[2][r][K][0] for u in us]); pt=np.array([u[2][r][P] for u in us]); c=np.array([u[2][r]['compression_vs_float32'] for u in us])
        print(f'{r:48s} comp~{np.median(c):6.1f} safe {np.mean(lo>=0.95):.2f} ({int((lo>=0.95).sum())}/{len(us)}) pt>=.97 {np.mean(pt>=.97):.2f} minpt {pt.min():.3f} medpt {np.median(pt):.3f}')
# per-unit best safe among the rung list (oracle) vs fixed ladder choice
print('--- fixed ladder vs oracle (rungs w/ comp>=1)')
ladders={'text':['float16','int8(per_vector)','int4(mean)','hierarchical_merge(0.5)'],}
for g in ['text','image128','image wide(>128)']:
    us=groups[g]
    cand=[r for r in rungs]
    for name,lad in {'A: int8':['int8(per_vector)'],'B: int4(mean)':['int4(mean)'],'C: merge.25>int8':['hierarchical_merge(0.25) > int8(per_vector)'],'D: merge.25>int4(mean)':['hierarchical_merge(0.25) > int4(mean)'],'E: merge.33>int4(mean)':['hierarchical_merge(0.33) > int4(mean)'],'F: merge.25>binary':['hierarchical_merge(0.25) > binary']}.items():
        r=lad[0]
        lo=np.array([u[2][r][K][0] for u in us]);c=np.array([u[2][r]['compression_vs_float32'] for u in us])
    orc=[];
    for u in us:
        best=max([u[2][r]['compression_vs_float32'] for r in cand if u[2][r][K][0]>=0.95],default=1)
        orc.append(best)
    print(g,'oracle (best safe of the rung list) median comp',np.median(orc),'mean',np.mean(orc).round(1))
# fallback ladder: pick most-compressed rung from ordered list that is safe in K% of units -> measure with leave-one-out? just report each rung's safe rate; and 'step down' ladder: try rung list in order from most compressed to least; if unsafe fall to next (needs calibrate) -- skip.
# failures of int4(mean) and int8
for g in ['text','image128','image wide(>128)']:
    for r in ['int8(per_vector)','int4(mean)','hierarchical_merge(0.25) > int8(per_vector)']:
        bad=[(o.split('--')[-1][:28],f.split('/')[-1][:30],round(rows[r][K][0],3),round(rows[r][P],3)) for f,o,rows in groups[g] if rows[r][K][0]<0.95]
        print(g,r,len(bad),bad[:12])
