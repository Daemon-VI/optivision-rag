from common import *
import pandas as pd
W=wide_files(); rec=[]
for (folder,fn),w in W.items():
    mid=folder.replace('--','/',1); b=w['baseline']
    for r in w['rows']:
        rec.append(dict(model=mid,fn=fn,label=r['label'],group=r['group'],vpd=r['vectors_per_doc'],dim=r['dim'],bpd=r['bits_per_dim'],bytes=r['bytes_per_doc'],comp=r['compression_vs_float32'],
          b_dim=b['dim'],b_vpd=b['vectors_per_doc'],b_f32=b['float32_bytes_per_doc'],b_f16=b['float16_bytes_per_doc']))
df=pd.DataFrame(rec)
df['comp_re']=df.b_f32/df.bytes
df['comp_err']=df.comp/df.comp_re-1
print("max |comp - f32/bytes|:",df.comp_err.abs().max())
df['bytes_naive']=df.vpd*df.dim*df.bpd/8
df['ovh']=df.bytes-df.bytes_naive
df['ovh_per_vec']=df.ovh/df.vpd
print(df.groupby('label').agg(n=('bpd','size'),bpd_min=('bpd','min'),bpd_max=('bpd','max'),ovh_vec_min=('ovh_per_vec','min'),ovh_vec_max=('ovh_per_vec','max')).to_string())
# float32 baseline row: bytes == b_f32 ?
f=df[df.label=='float32']; print("float32 row bytes vs baseline:",(f.bytes/f.b_f32-1).abs().max())
f=df[df.label=='float16']; print("float16 row:",(f.bytes/f.b_f16-1).abs().max(), f.comp.min(), f.comp.max())
# dtype column: for float16-inference models, is compression "vs float32" meaning true storage?
print(df[df.label=='float16'].groupby('model').comp.mean().head())
df.to_pickle(OUT+'/comp.pkl')
