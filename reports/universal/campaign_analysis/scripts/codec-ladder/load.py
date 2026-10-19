import json,glob,os,sys
import numpy as np, pandas as pd
R="C:/Users/Lenovo/projects/optivision-package"
CODECS=['float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary']
def load():
    rows=[]
    seen={}
    files=glob.glob(R+"/reports/universal/campaign/*/wide/*.json")+glob.glob(R+"/reports/universal/wide/*_test_subsampled.json")
    for f in files:
        d=json.load(open(f))
        mid=d['baseline']['encoder'].get('model') or os.path.basename(f)
        ds=d['dataset']
        key=(mid,ds)
        src='campaign' if 'campaign' in f else 'wide'
        b=d['baseline']
        enc=b['encoder']
        for r in d['rows']:
            if r['group']!='codec': continue
            ci=r['retention_ci:ndcg@5']
            rows.append(dict(model=mid,dataset=ds,src=src,file=f,codec=r['label'],ret=r['retention:ndcg@5'],lo=ci[0] if ci else np.nan,hi=ci[1] if ci else np.nan,
              ret_r1=r['retention:recall@1'],ret_r5=r['retention:recall@5'],ret_n10=r['retention:ndcg@10'],
              bytes=r['bytes_per_doc'],comp=r['compression_vs_float32'],dim=b['dim'],vpd=b['vectors_per_doc'],base=b['ndcg@5'],
              dtype=str(enc.get('dtype')),n_docs=d['n_docs'],n_q=d['n_queries']))
    return pd.DataFrame(rows)
if __name__=="__main__":
    df=load()
    print(df.groupby(['src']).model.nunique())
    print(df.model.nunique(), df.groupby(['model','dataset','src']).size().shape)
    dup=df.groupby(['model','dataset','codec']).src.nunique()
    print(dup[dup>1].head(20))
    print(df.codec.value_counts())
    print(df.dtype.value_counts()/7)
    print(df[df.codec=='binary'][['model','dataset','ret','dim','vpd','base']].sort_values('ret').to_string())
