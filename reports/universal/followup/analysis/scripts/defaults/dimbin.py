import json,collections,numpy as np
R='reports/universal/'
meta=json.load(open(R+'followup/analysis/scripts/defaults/units.json'))
K='retention_ci:ndcg@5'
by=collections.defaultdict(list)
for f,org,img,fol,dim in meta:
    rows={r['label']:r for r in json.load(open(f))['rows']}
    key=('img' if img else 'txt',dim)
    by[key].append((org,{l:rows[l][K][0] for l in ['binary','lloyd2(7)','int4','int4(mean)','hierarchical_merge(0.25) > binary','hierarchical_merge(0.25) > int4(mean)']},rows['binary']['retention:ndcg@5']))
for k in sorted(by):
    v=by[k]
    print(k,len(v),'units',len({o for o,_,_ in v}),'models; safe rate:',{l:round(np.mean([x[1][l]>=.95 for x in v]),2) for l in v[0][1]},'binary med pt',round(float(np.median([x[2] for x in v])),3))
# image128 binary failures by model
fails=collections.Counter(); tot=collections.Counter()
for o,d,_ in by[('img',128)]:
    tot[o]+=1; fails[o]+= d['binary']<.95
print({o.split('--')[-1][:26]:(fails[o],tot[o]) for o in tot if fails[o]})
print('models img128 total',len(tot),'with any binary failure',sum(1 for o in tot if fails[o]))
