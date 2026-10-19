import json,glob,re,numpy as np,pandas as pd
from scipy import stats
from prep import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
rows=[]
for f in glob.glob(R+"/reports/universal/campaign/*/wide/*.json")+glob.glob(R+"/reports/universal/wide/*_test_subsampled.json"):
    d=json.load(open(f)); mid=d['baseline']['encoder']['model']
    for r in d['rows']:
        m=re.match(r'project\((\d+)\)\s*>\s*(.*)',r['label'])
        if m and r['group']=='dim+codec': rows.append(dict(model=mid,ds=d['dataset'].split('_')[-3] if 'test' in d['dataset'] else 'scifact',k=int(m.group(1)),codec=m.group(2),ret=r['retention:ndcg@5'],dim=d['baseline']['dim'],bytes=r['bytes_per_doc']))
        if r['group']=='dim' : pass
p=pd.DataFrame(rows)
p['ds']=p.ds.replace({'':'x'})
print(p.groupby(['model']).k.unique())
pv=p.pivot_table(index=['model','ds','dim','k'],columns='codec',values='ret').reset_index()
pv['bin_pen']=pv['binary']/pv['float16']
pv['int4m_pen']=pv['int4(mean)']/pv['float16']
print(pv.round(4).to_string())
pv.to_csv('projbin.csv',index=False)
