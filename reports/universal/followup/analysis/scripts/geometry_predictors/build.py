import json,glob,os,pandas as pd
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
rows=[]
for g in sorted(glob.glob(R+'followup/f*/*/geometry/*.json')):
    p=g.replace(chr(92),'/').split('/')
    batch,model,tag=p[-4],p[-3],p[-1][:-5]
    G=json.load(open(g))
    wf=None
    cand=[R+f'followup/{batch}/{model}/wide/{tag}.json',R+f'campaign/{model}/wide/{tag}.json']+glob.glob(R+f'followup/f*/{model}/wide/{tag}.json')
    for c in cand:
        if os.path.exists(c): wf=c;break
    if wf is None: print('NOWIDE',batch,model,tag); continue
    W=json.load(open(wf)); ret={r['label']:r for r in W['rows']}
    base=dict(batch=batch,model=model.split('--')[1],tag=tag,wide=wf.split('/universal/')[1].split('/')[0],kind='text' if 'scifact' in tag or 'nfcorpus' in tag or 'fiqa' in tag else 'img',dim=G['dim'],
      cos_mean=G['doc_cos_to_mean']['mean'],cos_med=G['doc_cos_to_mean']['median'],p2r=G['peak_to_rms']['mean'],p2rc=G['peak_to_rms_centred']['mean'],
      sm1=G['second_moment_share']['top1'],sm4=G['second_moment_share']['top4'],maxabs=G['max_abs_coordinate_mean'],qcos=G['query_cos_to_doc_mean']['mean'],
      int4_sp=G['codecs']['int4']['score_pearson']['mean'],int4_vc=G['codecs']['int4']['vector_cosine']['mean'],
      int4m_sp=G['codecs']['int4(mean)']['score_pearson']['mean'],bin_sp=G['codecs']['binary']['score_pearson']['mean'],bin_vc=G['codecs']['binary']['vector_cosine']['mean'],
      binm_sp=G['codecs']['binary(mean)']['score_pearson']['mean'],int4_rr=G['codecs']['int4']['score_rel_rmse'])
    for lab,k in [('int4','int4'),('int4(mean)','int4m'),('binary','bin'),('hierarchical_merge(0.25) > int4','m25_int4'),('hierarchical_merge(0.25) > int4(mean)','m25_int4m'),('hierarchical_merge(0.25) > binary','m25_bin')]:
        if lab in ret:
            r=ret[lab]; d=dict(base);d['codec']=k;d['ret']=r['retention:ndcg@5'];d['lo']=r['retention_ci:ndcg@5'][0];rows.append(d)
df=pd.DataFrame(rows);df.to_csv('table.csv',index=False)
print(df.groupby('codec').size()); print(df[['batch','model','tag','wide']].drop_duplicates().to_string())
