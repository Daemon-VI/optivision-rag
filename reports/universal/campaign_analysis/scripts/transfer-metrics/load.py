import json,glob,os,pandas as pd,numpy as np
ROOT='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign'
def load():
    rows=[];bl=[]
    for f in sorted(glob.glob(ROOT+'/*/wide/*.json')):
        model=os.path.normpath(f).split(os.sep)[-3]
        d=json.load(open(f))
        ds='scifact' if 'scifact' in d['dataset'] else ('docvqa' if 'docvqa' in d['dataset'] else 'infovqa')
        b=d['baseline']
        bl.append(dict(model=model,ds=ds,nq=d['n_queries'],nd=d['n_docs'],base=b['ndcg@5'],base10=b['ndcg@10'],r1=b['recall@1'],r5=b['recall@5'],dtype=b['encoder'].get('dtype'),vpd=b['vectors_per_doc']))
        for r in d['rows']:
            x=dict(model=model,ds=ds,label=r['label'],group=r['group'],comp=r['compression_vs_float32'],vpd=r['vectors_per_doc'],dim=r['dim'],bits=r['bits_per_dim'])
            for m in ['ndcg@5','ndcg@10','recall@1','recall@5']:
                x['ret_'+m]=r['retention:'+m]; x['lo_'+m]=r['retention_ci:'+m][0]; x['hi_'+m]=r['retention_ci:'+m][1]; x['abs_'+m]=r[m]
            rows.append(x)
    return pd.DataFrame(rows),pd.DataFrame(bl)
if __name__=='__main__':
    R,B=load(); print(R.shape,B.shape); print(B.groupby('ds').agg(n=('model','count'),nq=('nq','mean'),base=('base','mean')))
    print(B['dtype'].value_counts())
    print(R.groupby('ds').group.value_counts())
    print((R['ret_ndcg@5']>1).mean(), R.comp.describe())
    # duplicate labels?
    print(R.duplicated(['model','ds','label']).sum())
