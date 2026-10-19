from common import *
import pandas as pd, glob
O=outcomes(); W=wide_files()
rows=[]
for mid,o in O.items():
    f=o['folder']; rj=CAMP+'/'+f+'/run.json'
    r=json.load(open(rj)) if os.path.exists(rj) else {}
    cr=r.get('check_retriever') or {}
    vpd=cr.get('vectors_per_document'); 
    bw={}
    for (fo,fn),w in W.items():
        if fo==f: bw[fn.split('_',1)[1][:7]]=w['baseline']
    rows.append(dict(model=mid,outcome=o['outcome'][:28],dtype=o['dtype'],multi=o['multi_gpu'],ok=o['check_retriever_ok'],
       rank1=cr.get('query_ranks_own_document_first'),pairs=cr.get('pairs'),dim=cr.get('dimension'),qdim=cr.get('query_dimension'),
       cr_vpd_mean=np.mean(vpd) if vpd else None,cr_vpd_min=min(vpd) if vpd else None,cr_vpd_max=max(vpd) if vpd else None,
       qv=sorted(set(cr.get('vectors_per_query') or [])),unit=cr.get('unit_norm_max_error'),
       nattempt=len(o.get('attempts',[]))+1,reg=r.get('registry_status'),
       b_vpd={k:round(v['vectors_per_doc'],1) for k,v in bw.items()}, b_dim=list({v['dim'] for v in bw.values()})))
df=pd.DataFrame(rows).sort_values('model'); pd.set_option('display.width',300); pd.set_option('display.max_colwidth',60)
print(df.drop(columns=['multi']).to_string())
df.to_pickle(OUT+'/models.pkl')
