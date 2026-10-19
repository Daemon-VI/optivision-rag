import json,glob,collections,numpy as np
R='reports/universal/'
units=[]
for f in glob.glob(R+'campaign/*/wide/*.json')+glob.glob(R+'followup/f[1-7]*/*/wide/*.json'):
    f=f.replace(chr(92),'/')
    d=json.load(open(f)); t=f.lower()
    img=any(k in t for k in ['docvqa','infovqa','arxivqa','tabfquad'])
    org=f.split('/')[-3]
    fol='followup' in f
    rows={r['label']:r for r in d['rows']}
    units.append(dict(f=f,org=org,img=img,fol=fol,dim=max(r.get('dim',0) or 0 for r in d['rows']),rows=rows,nd=d['n_docs']))
print(len(units),sum(u['img'] for u in units),sum(u['fol'] for u in units))
print('dims',collections.Counter((u['img'],u['dim']) for u in units))
labs=collections.Counter(l for u in units for l in u['rows'])
print([ (l,c) for l,c in labs.items() if c>=len(units)*0.9][:80])
json.dump([(u['f'],u['org'],u['img'],u['fol'],u['dim']) for u in units],open('reports/universal/followup/analysis/scripts/defaults/units.json','w'))
