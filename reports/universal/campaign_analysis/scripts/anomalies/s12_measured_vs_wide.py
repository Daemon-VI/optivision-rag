from common import *
import pandas as pd
M=measured(); W=wide_files()
def ds_key(fn): return 'BeIR/scifact' if 'scifact' in fn else ('vidore/docvqa_test_subsampled' if 'docvqa' in fn else 'vidore/infovqa_test_subsampled')
rec=[]; missing=[]
for (folder,fn),w in W.items():
    mid=folder.replace('--','/',1); ds=ds_key(fn); mm=M[mid][ds]
    metric=mm['metric']
    wl={r['label']:r for r in w['rows']}; ml={r['label']:r for r in mm['rows']}
    missing.append((mid,ds[-7:],len(wl),len(ml),sorted(set(wl)-set(ml))[:4]))
    for l,r in ml.items():
        x=wl[l]
        rec.append(dict(model=mid,ds=ds[-7:],label=l,d_ret=r['retention']-x['retention:'+metric],d_lo=r['retention_interval'][0]-x['retention_ci:'+metric][0],d_hi=r['retention_interval'][1]-x['retention_ci:'+metric][1],
          d_comp=r['compression']/x['compression_vs_float32']-1,d_vpd=r['vectors_per_doc']-x['vectors_per_doc'],d_bytes=r['bytes_per_doc']/x['bytes_per_doc']-1))
df=pd.DataFrame(rec)
print(df[['d_ret','d_lo','d_hi','d_comp','d_vpd','d_bytes']].abs().max())
print("rows in wide but not in measured.json:")
tot=0
for m in missing:
    if m[2]!=m[3]: print(m); tot+=m[2]-m[3]
print('total dropped rows',tot, 'wide rows total',sum(m[2] for m in missing))
