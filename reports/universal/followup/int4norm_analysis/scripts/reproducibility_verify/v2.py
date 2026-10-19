import json,glob,sys
from scipy.stats import spearmanr,kendalltau
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup'
def load(b,ds):
    o={}
    for p in glob.glob(f'{R}/{b}/*/int4norm/*_{ds}.json'):
        p=p.replace(chr(92),'/'); m=p.split(b+'/')[1].split('/')[0]
        o[m.split('--')[1]]=json.load(open(p))
    return o
D={'scifact':load('f9b_int4norm_text','scifact'),'nfcorpus':load('f9b_int4norm_text','nfcorpus'),'scidocs':load('f12b_text_scidocs','scidocs')}
for k,v in D.items():
    x=next(iter(v.values())); print(k,len(v),x['n_docs'],x['n_queries'])
def ret(d,l):
    r={x['label']:x for x in d['rows']}[l]; return r['retention:ndcg@5'],r.get('retention_ci:ndcg@5')
for l in ['int4','int4norm','int4(mean)','int4norm(mean)']:
    print('==',l)
    ms=sorted(set(D['scifact'])|set(D['scidocs']))
    for m in ms:
        print(m.ljust(36),*[('%.3f'%ret(D[k][m],l)[0] if m in D[k] else '  -  ') for k in D], ret(D['scidocs'][m],l)[1] if m in D['scidocs'] else '')
    for a,b in [('scifact','nfcorpus'),('scifact','scidocs'),('nfcorpus','scidocs')]:
        c=[m for m in D[a] if m in D[b]]
        print(a,b,len(c),'sp %.2f'%spearmanr([ret(D[a][m],l)[0] for m in c],[ret(D[b][m],l)[0] for m in c])[0])
# broken set
for k in D:
    print(k,sorted(m for m in D[k] if ret(D[k][m],'int4')[0]<0.90))
# int4norm(mean) range
v=[(ret(D[k][m],'int4norm(mean)')[0],k,m) for k in D for m in D[k]]
print(len(v),min(v),max(v))
for k in D: print(k,min((ret(D[k][m],'int4norm(mean)')[0],m) for m in D[k]),min((ret(D[k][m],'int4norm')[0],m) for m in D[k]))
