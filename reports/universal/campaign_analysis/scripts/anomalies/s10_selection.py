from common import *
import pandas as pd, glob
rec=[]
for f in glob.glob(CAMP+'/*/selection_rules/*.json'):
    folder=f.replace(os.sep,'/').split('/')[-3]; d=json.load(open(f))
    for r in d['results']:
        rec.append(dict(model=folder.replace('--','/',1),ds=os.path.basename(f).split('_',1)[1][:7],ref=r['reference'],rule=r['rule'],target=r['target'],
          selected=r['selected'],splits=r['splits'],met=r['met'],met_labels=r['met_labels'],hm=r['holdout_mean'],hmin=r['holdout_min'],comp=r['compression_median'],
          nq=d['n_queries'],ncand=d['n_candidates'],metric=d['metric']))
df=pd.DataFrame(rec); print(len(df),'result rows from',df.groupby(['model','ds']).ngroups,'files; metrics',df.metric.unique())
print(df.groupby('ref').size().to_dict())
pd.set_option('display.width',250)
g=df.groupby(['ref','rule','target']).agg(n=('met','size'),met_mean=('met','mean'),met_labels_mean=('met_labels','mean'),frac_lt_nom=('met',lambda s:(s<0.9).mean()),hmin_min=('hmin','min'),comp=('comp','median')).round(3)
print(g.to_string())
# compare met and met_labels
print("met != met_labels rows:",(abs(df.met-df.met_labels)>1e-9).sum(), "of", len(df))
# splits constant?
print(df.splits.value_counts().to_dict())
df.to_pickle(OUT+'/sel.pkl')
