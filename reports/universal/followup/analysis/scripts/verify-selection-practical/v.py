import json,glob,collections,numpy as np
base='reports/universal/followup/'
units=[f.replace(chr(92),'/') for f in glob.glob(base+'f[1-7]*/*/selection_rules/*.json') if 'f3b' not in f]
img=lambda f:any(k in f.lower() for k in ['docvqa','infovqa','arxivqa','tabfquad'])
mer=collections.defaultdict(lambda:[0,0]); pairs=collections.Counter(); fl=0
for f in units:
    m='img' if img(f) else 'txt'
    d=json.load(open(f))
    by={}
    for r in d['results']:
        if r['reference']!='labels': continue
        by[(r['rule'],r['target'])]=r
        if r['rule']=='calibrate() default':
            for o in r['outcomes']:
                s=o['selected']; mer[(m,r['target'])][0]+= s is not None and ('merge' in s); mer[(m,r['target'])][1]+=1
    for t in (.95,.97,.99):
        a=by[('calibrate() default',t)];b=by[('lower 0.999',t)]
        pairs['n']+=1; pairs['lower']+= a['met']<b['met']; pairs['eq']+=a['met']==b['met']
        # compare default vs l999 met per split
        ao=a['outcomes'];bo=b['outcomes']
        pairs['splits']+=len(ao)
        pairs['split_def_worse']+=sum(1 for x,y in zip(ao,bo) if y.get('met',None) is True and x.get('met',None) is False) if 'met' in ao[0] else 0
print(pairs); print({k:(v[0],v[1],round(v[0]/v[1],2)) for k,v in sorted(mer.items())})
print(list(json.load(open(units[0]))['results'][0]['outcomes'][0].keys()), list(json.load(open(units[0]))['results'][0].keys()))
# oracle: best compression with point retention>=target on full data, for text units vs default median
for f in units:
    if img(f): continue
    w=f.replace('selection_rules','wide'); d=json.load(open(w))
    r97=[x for x in json.load(open(f))['results'] if x['rule']=='calibrate() default' and x['target']==.97 and x['reference']=='labels'][0]
    best=max(x['compression_vs_float32'] for x in d['rows'] if x['retention:ndcg@5']>=.97 and x['retention_ci:ndcg@5'][0]>=.97*0+.95)
    best2=max(x['compression_vs_float32'] for x in d['rows'] if x['retention_ci:ndcg@5'][0]>=.97)
    print(f.split('/')[-1][:40], 'default',round(r97['compression_median'],1),'oracle pt>=.97&lo>=.95',round(best,1),'oracle lo>=.97',round(best2,1))
