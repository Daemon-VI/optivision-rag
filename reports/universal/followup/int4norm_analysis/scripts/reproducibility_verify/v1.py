import json,glob,os,re,collections
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup'
C='C:/Users/Lenovo/projects/optivision-package/reports/universal'
def files(base):
    out={}
    for p in glob.glob(base+'/**/*.json',recursive=True):
        p=p.replace(chr(92),'/')
        if '/geometry/' in p or 'run.json' in p or 'config' in os.path.basename(p): continue
        try: d=json.load(open(p))
        except: continue
        if not isinstance(d,dict) or 'rows' not in d: continue
        out[p]=d
    return out
s17={}
for b in ['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']:
    for p,d in files(f'{R}/{b}').items():
        m=p.split(b+'/')[1].split('/')[0]
        s17[(b,m,d['dataset'])]=d
print('s17 units',len(s17))
old=collections.defaultdict(list)
for b in glob.glob(R+'/f*_*')+[C+'/campaign']:
    b=b.replace(chr(92),'/')
    if re.search(r'f\d+b_',b) and 'f3b' not in b: continue
    for p,d in files(b).items():
        if '/int4norm/' in p and False: continue
        rel=p.split(b+'/')[1]; m=rel.split('/')[0]
        old[(m,d['dataset'])].append((b.split('/')[-1],p,d))
print('old units',len(old))
def rows(d): return {r['label']:r for r in d['rows']}
nt=0;nm=0;mx=0;maxb=0;units=set();unm=[];fields=collections.Counter()
for k,d in s17.items():
    key=(k[1],k[2]); 
    if key not in old: unm.append(k);continue
    for b,p,o in old[key]:
        if o['n_docs']!=d['n_docs']:
            continue
        nt+=1;units.add(k)
        rs,ro=rows(d),rows(o)
        sh=set(rs)&set(ro)
        for l in sh:
            for f,v in rs[l].items():
                if isinstance(v,(int,float)) and f in ro[l] and isinstance(ro[l][f],(int,float)) and 'seconds' not in f and 'ms' not in f:
                    dd=abs(v-ro[l][f]); mx=max(mx,dd)
                    if dd>0: fields[(l,f)]+=1
        for f in ['ndcg@5','ndcg@10']:
            maxb=max(maxb,abs(d['baseline'][f]-o['baseline'][f]))
        nm+=len(sh)
print('comparisons',nt,'units',len(units),'shared rows',nm,'max diff',mx,'maxbase',maxb,fields.most_common(5))
print('unmatched',len(unm));[print(u) for u in unm]
print(sorted(set(r for k in s17 for r in rows(s17[k])))[:30])
print('---detail')
for k,d in s17.items():
    key=(k[1],k[2])
    for b,p,o in old.get(key,[]):
        if o['n_docs']!=d['n_docs']: continue
        rs,ro=rows(d),rows(o); bad=[]
        for l in set(rs)&set(ro):
            for f in ['ndcg@5','ndcg@10','bytes_per_doc','retention:ndcg@5']:
                if f in rs[l] and f in ro[l] and rs[l][f]!=ro[l][f]: bad.append((l,f,rs[l][f],ro[l][f]))
        if bad: print(k,b,p.split('/')[-3:],len(bad),bad[:2])
