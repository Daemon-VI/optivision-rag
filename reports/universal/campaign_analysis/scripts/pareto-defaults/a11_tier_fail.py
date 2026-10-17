from load import *
w,t,b=wide_all()
for h,g in t.groupby('hot'):
    f=g[g.lo<.95].sort_values('lo'); print(h,len(f)); print(f[['key','ret','lo','ramc','diskc']].round(3).to_string(index=False))
# compare int8 alone vs tiered binary hot on same pairs (cold int8 is upper bound-ish)
i8=w[w.label=='int8(per_vector)'].set_index('key'); tb=t[t.hot=='binary'].set_index('key'); j=tb.join(i8,lsuffix='_t',rsuffix='_i8')
print("tiered-binary ret - int8 ret: mean %.4f min %.4f"%((j.ret_t-j.ret_i8).mean(),(j.ret_t-j.ret_i8).min()))
# storage totals: bytes per doc (MB) for a typical 500-page image corpus
print("median float32 MB/page image:",b[b.ds!='scifact'].f32.median()/1e6)
