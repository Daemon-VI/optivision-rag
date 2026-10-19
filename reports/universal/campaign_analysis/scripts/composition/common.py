import json,re,numpy as np,pandas as pd
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
def load():
    recs=[]
    for mid,dss in M.items():
        for ds,d in dss.items():
            rows={r['label']:r for r in d['rows']}
            for l,r in rows.items():
                st=[s for s in l.split(' > ')]
                recs.append(dict(model=mid,ds=ds.split('/')[-1].split('_')[0],label=l,stages=tuple(st),ret=r['retention'],lo=r['retention_interval'][0],hi=r['retention_interval'][1],comp=r['compression'],vpd=r['vectors_per_doc'],base=d['baseline'],nq=d['n_queries'],note=d.get('note','')))
    return pd.DataFrame(recs)

import re
CODECS=['float16','int8(per_vector)','int4','int4(mean)','lloyd2(7)','binary']
def kind(s):
    if s in CODECS: return 'codec'
    if s.startswith('hierarchical_merge'): return 'merge'
    if s.startswith('project('): return 'project'
    return 'other'
def build_pairs():
    """rows for every canonical combined pipeline whose parts have singles / sub-pipelines measured"""
    df=load()
    df=df[~df.label.str.contains('ward|adaptive r|none >|int8/vec|\[|random|adaptive_merge')]
    idx={(r.model,r.ds,r.label):r for r in df.itertuples()}
    out=[]
    for r in df[df.stages.map(len)>1].itertuples():
        st=r.stages
        kinds=tuple(kind(s) for s in st)
        key=lambda lab: idx.get((r.model,r.ds,lab))
        singles=[key(s) for s in st]
        base=dict(model=r.model,ds=r.ds,label=r.label,kinds='>'.join(kinds),nst=len(st),act=r.ret,alo=r.lo,ahi=r.hi,comp=r.comp,note=r.note,stages=st)
        if all(singles):
            p=np.prod([s.ret for s in singles])
            out.append({**base,'pred_type':'all_singles','pred':p,'parts':[s.ret for s in singles],'plo':[s.lo for s in singles]})
        if len(st)==3:
            a=key(' > '.join(st[:2])); b=key(' > '.join(st[1:])); 
            if a and key(st[2]): out.append({**base,'pred_type':'(m>p)x codec','pred':a.ret*key(st[2]).ret,'parts':[a.ret,key(st[2]).ret],'plo':[]})
            if b and key(st[0]): out.append({**base,'pred_type':'merge x (p>c)','pred':b.ret*key(st[0]).ret,'parts':[key(st[0]).ret,b.ret],'plo':[]})
    return pd.DataFrame(out)
