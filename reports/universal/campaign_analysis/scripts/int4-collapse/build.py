import json,re,sys
import pandas as pd
from optivision.adapters import MODELS
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
rows=[]
for m,ds in M.items():
    mi=MODELS.get(m) if hasattr(MODELS,'get') else None
    for d,v in ds.items():
        r={ 'model':m,'dataset':d.split('/')[-1],'baseline':v['baseline'],'n_docs':v['n_docs'],'n_q':v['n_queries'],'note':v.get('note')}
        for x in v['rows']:
            if len(x['pipeline']['stages'])<=1:
                r[x['label']]=x['retention']; r[x['label']+'_lo']=x['retention_interval'][0]; r[x['label']+'_hi']=x['retention_interval'][1]
                if x['label']=='float32': r['vpd']=x['vectors_per_doc']; r['bytes32']=x['bytes_per_doc']
        if mi is not None:
            r.update(family=mi.family,modality=mi.modality,dim=mi.dimension,status=mi.status,evidence=mi.evidence,license=mi.license,notes=mi.notes,revision=mi.revision)
        rows.append(r)
df=pd.DataFrame(rows)
df.to_pickle('master.pkl')
print(type(MODELS),len(MODELS))
pd.set_option('display.width',250,'display.max_rows',200,'display.max_colwidth',40)
print(df[['model','dataset','family','modality','dim','baseline','vpd','int4','int4(mean)','binary','int8(per_vector)']].round(3).to_string())
