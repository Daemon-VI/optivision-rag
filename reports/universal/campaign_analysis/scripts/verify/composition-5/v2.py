exec(open('v1.py').read().split("def auc")[0])
A['err']=A.act-A.pred; A['hw']=(A.ahi-A.alo)/2
print('err mean',A.err.mean(),'sd',A.err.std(),'median hw',A.hw.median())
# near-T mean error is biased? bias of pred in band
for T in [.95,.98]:
    b=A[(A.pred>=T-.02)&(A.pred<T+.02)];print(T,'band n',len(b),'mean err',b.err.mean(),'frac act<pred',(b.err<0).mean(),'mean z',(b.err/(b.hw/1.96)).mean())
# per-model mean err (systematic model effect)
pm=A.groupby('model').err.mean();print('per-model mean err sd',pm.std(),'range',pm.min(),pm.max())
# does err beyond CI-noise? share |err|>hw
print('share |err|>hw',(A.err.abs()>A.hw).mean())
# paired-noise note: act and pred share same queries so errs are correlated; skip
# model-level precision for 0.98 over 51 models: median
