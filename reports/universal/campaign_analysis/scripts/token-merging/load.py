import json,re,sys
import pandas as pd, numpy as np
sys.path.insert(0,'C:/Users/Lenovo/projects/optivision-package/src')
from optivision.adapters import MODELS
P='C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'
def load():
    d=json.load(open(P))['models']
    recs=[]
    for m,dd in d.items():
        mi=MODELS.get(m) if hasattr(MODELS,'get') else None
        for ds,v in dd.items():
            for r in v['rows']:
                st=r['pipeline']['stages']
                if len(st)==1 and st[0]['stage'] in('hierarchical_merge','adaptive_merge'):
                    s=st[0]
                    recs.append(dict(model=m,ds=ds.split('/')[-1].split('_')[0],n_docs=v['n_docs'],n_q=v['n_queries'],
                      stage=s['stage'],ratio=s.get('ratio'),radius=s.get('radius'),ret=r['retention'],lo=r['retention_interval'][0],hi=r['retention_interval'][1],
                      vpd=r['vectors_per_doc'],comp=r['compression'],label=r['label'],base=v['baseline']))
    df=pd.DataFrame(recs)
    return df,d
if __name__=='__main__':
    df,d=load()
    print(df.shape,df.model.nunique());print(df.groupby(['stage','ratio','radius'],dropna=False).size())
    print(type(MODELS)); k=list(MODELS)[0]; print(MODELS[k])
    print(df.head())
    print(df.groupby('ds').model.nunique())
