"""A6: does lowering dimension change how much binary/int4 codecs cost? ret(project>codec)/ret(project) vs ret(codec)."""
from common import *
pd.set_option('display.width',250,'display.max_rows',200)
al=df.copy()
base_codec={}
for c in ['binary','int8','int4','float16']:
    pass
cod=df[df.names.isin(['binary','int4','int8'])&df.label.isin(['binary','int4(mean)','int8(per_vector)'])].copy()
print(cod.label.value_counts())
cod['c']=cod.names
cmap=cod.set_index(['model','ds','c']).ret.to_dict()
x=df[df.names.isin(['project>binary','project>int4','project>int8'])&(df.method!='random')].copy()
x['c']=x.names.str.split('>').str[1]
# keep int4(mean) only
x=x[(x.c!='int4')|x.label.str.contains('int4\(mean\)')]
x['codec_alone']=[cmap.get((m,d,c),np.nan) for m,d,c in zip(x.model,x.ds,x.c)]
pr=curve.set_index(['model','ds','k'])
prj=pp[pp.method=='pca'].set_index(['model','ds','k']).ret
x['proj_alone']=[prj.get((m,d,k),np.nan) for m,d,k in zip(x.model,x.ds,x.k)]
x['marg']=x.ret/x.proj_alone          # cost of codec after projection
x['codec_cost_full']=x.codec_alone    # cost of codec at full dim
x['excess']=x.marg-x.codec_alone
print(x.groupby(['c','D','k']).agg(n=('ret','size'),codec_full=('codec_alone','mean'),codec_after_proj=('marg','mean'),excess=('excess','mean')).round(4).to_string())
for c,g in x.groupby('c'):
    for lab,gg in [('128-d k=64',g[g.D==128]),('wide',g[g.D>128])]:
        e,lo,hi=cluster_boot(gg.excess,gg.model,B=3000);print(c,lab,'n=%d excess (after-proj cost minus full-dim cost) mean %.4f CI [%.4f,%.4f]'%(len(gg),e,lo,hi))

print('\nBy model, binary/int4 excess (after-PCA codec cost minus codec alone), mean over k and ds, wide only')
g=x[x.D>128].groupby(['D','model','c']).excess.mean().unstack().round(4);print(g.to_string())
for c_ in ('binary','int4'):
    a=x[(x.D>=2048)&(x.c==c_)];b=x[(x.D<=640)&(x.c==c_)]
    print(c_,'D>=2048: n=%d models=%d mean excess %.4f CI'%(len(a),a.model.nunique(),a.excess.mean()),np.round(cluster_boot(a.excess,a.model,B=3000)[1:],4),' | D<=640: n=%d mean %.4f CI'%(len(b),b.excess.mean()),np.round(cluster_boot(b.excess,b.model,B=3000)[1:],4))
