from load import *
import glob,os
df,d=load()
# baseline vpd from wide files (campaign + older)
bv={}
for f in glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/**/*.json',recursive=True):
    if 'selection' in f or 'run.json' in f or 'outcomes' in f: continue
    try: j=json.load(open(f))
    except: continue
    if isinstance(j,dict) and 'baseline' in j and 'rows' in j and isinstance(j['baseline'],dict) and 'vectors_per_doc' in j['baseline']:
        ds=[k for k in ('docvqa','infovqa','scifact') if k in j['dataset'].lower() or k in f.lower()]
        if not ds: continue
        ds=ds[0]
        parts=os.path.normpath(f).split(os.sep)
        if 'campaign' in parts:
            m=parts[parts.index('campaign')+1].replace('--','/',1)
            bv[(m,ds)]=j['baseline']['vectors_per_doc']
print(len(bv))
rows=[]
for (m,ds),g in df.groupby(['model','ds']):
    rec=dict(model=m,ds=ds,base=g.base.iloc[0],n_q=g.n_q.iloc[0])
    for rt in [0.5,0.33,0.25,0.1]:
        x=g[(g.stage=='hierarchical_merge')&(g.ratio==rt)]
        if len(x):
            x=x.iloc[0];rec[f'r{rt}']=x.ret;rec[f'lo{rt}']=x.lo;rec[f'hi{rt}']=x.hi;rec[f'v{rt}']=x.vpd
    x=g[(g.stage=='adaptive_merge')&(g.radius==0.6)]
    if len(x): rec['a_ret']=x.iloc[0].ret;rec['a_lo']=x.iloc[0].lo;rec['a_hi']=x.iloc[0].hi;rec['a_v']=x.iloc[0].vpd
    rec['bvpd']=bv.get((m,ds),np.nan)
    mi=MODELS[m];rec['family']=mi.family;rec['modality']=mi.modality;rec['dim']=mi.dimension;rec['status']=mi.status;rec['ev']=mi.evidence
    rows.append(rec)
W=pd.DataFrame(rows)
W['bvpd_est']=W['v0.5']/0.5
print(W.bvpd.isna().sum(), (W.bvpd/W.bvpd_est).describe())
W['bv']=W.bvpd.fillna(W.bvpd_est)
W['f16']=W.ev.str.contains('float16|bfloat16|fp16|bf16',case=False)
W.to_pickle('W.pkl')
pd.set_option('display.width',250,'display.max_columns',40,'display.max_rows',200)
print(W[['model','ds','family','dim','bv','base','r0.5','r0.33','r0.25','r0.1','a_ret','a_v']].round(3).to_string())
print(W.ev.head(3).tolist())
