import json,glob,os,csv
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
rows=[]
for b in ['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']:
    for f in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
        d=json.load(open(f)); model=os.path.normpath(f).replace(os.sep,'/').split('/')[-3]
        gf=os.path.normpath(f).replace(os.sep+'int4norm'+os.sep,os.sep+'geometry'+os.sep)
        g=json.load(open(gf)) if os.path.exists(gf) else None
        if g and 'second_moment_share' not in g: print('BAD',gf,list(g)[:8]); g=None
        r={x['label']:x for x in d['rows']}
        def ret(l): return r[l]['retention:ndcg@5'] if l in r else None
        def lo(l): return r[l]['retention_ci:ndcg@5'][0] if l in r else None
        o=dict(batch=b,model=model,ds=d['dataset'],nq=d['n_queries'],nd=d['n_docs'],dim=d['baseline']['dim'],
          i4=ret('int4'),i4n=ret('int4norm'),m4=ret('int4(mean)'),m4n=ret('int4norm(mean)'),i8=ret('int8(per_vector)'),
          i4_lo=lo('int4'),i4n_lo=lo('int4norm'),m4_lo=lo('int4(mean)'),m4n_lo=lo('int4norm(mean)'))
        n=d['norms']
        for c in ['int4','int4norm','int4(mean)','int4norm(mean)']:
            o['sd_'+c]=n[c]['ratio_sd']; o['mean_'+c]=n[c]['ratio_mean']
        if g:
            o.update(top1=g['second_moment_share']['top1'],top4=g['second_moment_share']['top4'],p2r=g['peak_to_rms']['mean'],p2rc=g['peak_to_rms_centred']['mean'],
              cosm=g['doc_cos_to_mean']['mean'],mvn=g['mean_vector_norm'],maxabs=g['max_abs_coordinate_mean'],qcos=g['query_cos_to_doc_mean']['mean'],
              cos4=g['codecs']['int4']['vector_cosine']['mean'],pe4=g['codecs']['int4']['score_pearson']['mean'],pe4m=g['codecs']['int4(mean)']['score_pearson']['mean'])
        rows.append(o)
keys=sorted({k for o in rows for k in o},key=lambda k:list(rows[0]).index(k) if k in rows[0] else 99)
with open('units.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,keys);w.writeheader();w.writerows(rows)
print(len(rows),sum('top1' in o for o in rows))
for o in rows: print(o['batch'][:4],o['model'][:28],o['ds'][-18:],o['i4'] and round(o['i4'],3),round(o['i4n'],3),round(o['m4'],3),round(o['m4n'],3),'sd',round(o['sd_int4'],4),'top4',o.get('top4') and round(o['top4'],2),'cosm',o.get('cosm') and round(o['cosm'],2))
