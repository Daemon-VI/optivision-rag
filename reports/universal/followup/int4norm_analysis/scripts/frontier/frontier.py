import json,glob,collections,statistics as st
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
fams={'text(f9b)':['f9b_int4norm_text'],'pages(f10b+f11b)':['f10b_int4norm_pages_128d','f11b_int4norm_pages_wide'],'scidocs(f12b)':['f12b_text_scidocs'],'scale(f13b)':['f13b_text_scale_same_domain']}
K='retention:ndcg@5';C='retention_ci:ndcg@5'
def load(fam):
    u=[]
    for b in fams[fam]:
        for f in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
            d=json.load(open(f));u.append((f.split('/')[-1],d))
    return u
M=lambda x:'merge'+x
def short(l):return l.replace('hierarchical_merge','hm').replace('(per_vector)','')
def pareto(rows):
    out=[]
    for r in rows:
        dom=any((o['bytes_per_doc']<=r['bytes_per_doc']+1 and o[K]>=r[K] and (o['bytes_per_doc']<r['bytes_per_doc']-1 or o[K]>r[K])) for o in rows if o is not r)
        if not dom: out.append(r['label'])
    return out
for fam in fams:
    U=load(fam);n=len(U);print('\n=====',fam,'units',n,'queries',sorted(set(d['n_queries'] for _,d in U)))
    labs=[r['label'] for r in U[0][1]['rows']]
    fc=collections.Counter();pt=collections.Counter();lb=collections.Counter();mean=collections.defaultdict(list);byt=collections.defaultdict(list)
    for name,d in U:
        rows=d['rows'];assert [r['label'] for r in rows]==labs
        for l in pareto(rows):fc[l]+=1
        for r in rows:
            mean[r['label']].append(r[K]);byt[r['label']].append(r['bytes_per_doc']/[x for x in rows if x['label']=='float32'][0]['bytes_per_doc'])
            if r[K]>=.95:pt[r['label']]+=1
            if r[C][0]>=.95:lb[r['label']]+=1
    print('label | frontier | pt>=.95 | lb>=.95 | mean ret | median ret | min | bytes/f32')
    for l in labs:
        v=mean[l];print(f'{short(l)} | {fc[l]} | {pt[l]} | {lb[l]} | {st.mean(v):.3f} | {st.median(v):.3f} | {min(v):.3f} | {st.mean(byt[l]):.4f}')
    # pairwise
    def pair(a,b):
        A=[next(r for r in d['rows'] if r['label']==a) for _,d in U];B=[next(r for r in d['rows'] if r['label']==b) for _,d in U]
        diff=[x[K]-y[K] for x,y in zip(A,B)]
        w=sum(x>1e-9 for x in diff);l=sum(x<-1e-9 for x in diff)
        # CI separation: A lb > B ub
        sep=sum(x[C][0]>y[C][1] for x,y in zip(A,B));sepr=sum(y[C][0]>x[C][1] for x,y in zip(A,B))
        print(f'{short(a)} - {short(b)}: mean diff {st.mean(diff):+.4f} median {st.median(diff):+.4f} A>B {w} A<B {l} (CI-separated A {sep}, B {sepr}) bytes ratio {st.mean(x["bytes_per_doc"]/y["bytes_per_doc"] for x,y in zip(A,B)):.3f}')
    print('-- pairs')
    for a,b in [('int4norm(mean)','int8(per_vector)'),('int4norm(mean)','int4(mean)'),('int4norm(mean)','int4norm'),('int4norm(mean)','lloyd2(7)'),('int4norm(mean)','binary'),
        ('hierarchical_merge(0.33) > int4norm(mean)','hierarchical_merge(0.33) > int4(mean)'),('hierarchical_merge(0.33) > int4norm(mean)','hierarchical_merge(0.33) > int8(per_vector)'),('hierarchical_merge(0.33) > int4(mean)','hierarchical_merge(0.33) > int8(per_vector)'),
        ('hierarchical_merge(0.25) > int4norm(mean)','hierarchical_merge(0.25) > int4(mean)'),('hierarchical_merge(0.25) > int4norm(mean)','hierarchical_merge(0.25) > int8(per_vector)'),('hierarchical_merge(0.25) > int4(mean)','hierarchical_merge(0.25) > int8(per_vector)'),
        ('hierarchical_merge(0.33) > int4norm','hierarchical_merge(0.33) > int4'),('hierarchical_merge(0.25) > int4norm','hierarchical_merge(0.25) > int4'),
        ('hierarchical_merge(0.33) > int4norm(mean)','int4norm(mean)'),('hierarchical_merge(0.33) > int8(per_vector)','int4norm(mean)'),('lloyd2(7)','hierarchical_merge(0.33) > int4norm(mean)'),('lloyd2(7)','hierarchical_merge(0.33) > int8(per_vector)')]:
        pair(a,b)
    # per-unit lloyd2 vs int4norm(mean) retention, merges retention comparison at similar bytes
    if fam=='text(f9b)' or True:
        print('-- per unit (name: i4nm, i8, lloyd2, bin, m33>i4nm(mean), m33>i8, m25>i4nm(mean), m25>i8, m25 bytes ratio vs i4nm)')
        for name,d in U:
            g={r['label']:r for r in d['rows']}
            print(name[:42],*[f"{g[l][K]:.3f}" for l in ['int4norm(mean)','int8(per_vector)','lloyd2(7)','binary','hierarchical_merge(0.33) > int4norm(mean)','hierarchical_merge(0.33) > int8(per_vector)','hierarchical_merge(0.25) > int4norm(mean)','hierarchical_merge(0.25) > int8(per_vector)']],f"{g['hierarchical_merge(0.25) > int4norm(mean)']['bytes_per_doc']/g['int4norm(mean)']['bytes_per_doc']:.3f}")
