import json,glob,collections,numpy as np
base='reports/universal/followup/'
units=[f.replace(chr(92),'/') for f in glob.glob(base+'f[1-7]*/*/selection_rules/*.json') if 'f3b' not in f]
def img(f): return any(k in f.lower() for k in ['docvqa','infovqa','arxivqa','tabfquad'])
rows=[]
for f in units:
    d=json.load(open(f)); name=f.split('/')[-1][:-5]
    for r in d['results']:
        if r['reference']=='labels':
            rows.append((name,img(f),r['rule'],r['target'],r['met'],r['compression_median'],r['splits'],r['holdout_min']))
# pooled met per rule/target over all 22 units (each 20 splits)
print('pooled met over 22 units x 20 splits; and units with met<0.9')
for rule in ['calibrate() default','lower 0.999','bonferroni 0.975','lower 0.975 + margin 0.01','lower 0.99','lower 0.975','point']:
    s=[]
    for t in [0.95,0.97,0.99]:
        m=[r[4] for r in rows if r[2]==rule and r[3]==t]
        s.append(f'{t}: {np.mean(m):.3f} (units<0.9: {sum(x<0.9 for x in m)})')
    print(f'{rule:28s}',' | '.join(s))
print('default units with met<0.9')
for r in rows:
    if r[2]=='calibrate() default' and r[4]<0.9: print(r[0],r[1],r[3],r[4],r[5],'holdmin',round(r[7],3))
print('lower 0.999 units met<0.9')
for r in rows:
    if r[2]=='lower 0.999' and r[4]<0.9: print(r[0],r[1],r[3],r[4],r[5])
# paired met default vs l999
print('paired: default - l999 met (mean per modality,target) and pooled-split counts')
for m in [False,True]:
  for t in [.95,.97,.99]:
    a={r[0]:r for r in rows if r[2]=='calibrate() default' and r[3]==t and r[1]==m}
    b={r[0]:r for r in rows if r[2]=='lower 0.999' and r[3]==t and r[1]==m}
    dm=np.mean([a[k][4]-b[k][4] for k in a]); 
    print('image' if m else 'text',t,len(a),'met diff',round(dm,3),'default higher in',sum(a[k][4]>b[k][4] for k in a),'lower in',sum(a[k][4]<b[k][4] for k in a))
# per-unit default compression median by unit, target .97
print('default comp median per unit @0.97')
for r in sorted(rows,key=lambda x:(x[1],x[0])):
    if r[2]=='calibrate() default' and r[3]==0.97: print(r[0][:40],'img' if r[1] else 'txt',r[5],r[4])
