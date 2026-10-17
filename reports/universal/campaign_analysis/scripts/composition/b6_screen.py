import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
from scipy.stats import spearmanr
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P.copy(); A['err']=A.act-A.pred; A['loss']=1-A.pred
A['lbin']=pd.cut(A.loss,[-1,0.01,0.03,0.07,0.15,0.3,1.01])
print('=== error by predicted-loss bin')
for pt in ['all_singles','(m>p)x codec']:
    g=A[A.pred_type==pt]
    print(pt);print(g.groupby('lbin').apply(lambda x:pd.Series(dict(n=len(x),MAE=x.err.abs().mean(),bias=x.err.mean(),relMAE=(x.err.abs()/np.maximum(x.loss,0.005)).median(),min=x.err.min(),max=x.err.max()))).round(4))
print('=== threshold screening (all_singles & (m>p)xcodec), actual>=T vs pred>=T (+margin)')
rows=[]
for pt in ['all_singles','(m>p)x codec']:
  g=A[A.pred_type==pt]
  for T in [0.90,0.95,0.98,0.99]:
    for mar in [0,0.01,0.02]:
        pos=g.act>=T; sel=g.pred>=T+mar
        tp=(pos&sel).sum(); fp=(~pos&sel).sum(); fn=(pos&~sel).sum()
        rows.append(dict(pt=pt,T=T,margin=mar,n=len(g),base_rate=pos.mean(),prec=tp/max(1,tp+fp),rec=tp/max(1,tp+fn),fp=fp,fn=fn,fp_rate_of_sel=fp/max(1,sel.sum())))
print(pd.DataFrame(rows).round(3).to_string())
# FP severity: among false positives, how far below T
for pt in ['all_singles']:
  g=A[A.pred_type==pt]
  for T in [0.95,0.98]:
    fp=g[(g.act<T)&(g.pred>=T)]
    print(pt,T,'FP n',len(fp),'shortfall mean %.4f max %.4f'%((T-fp.act).mean(),(T-fp.act).max()))
print('=== within-model-ds rank correlation of pred vs act (all_singles, rows with pred<0.995)')
r=[]
for (m,d),g in A[A.pred_type=='all_singles'].groupby(['model','ds']):
    g=g[g.loss>0.0]
    if len(g)>=10: r.append(spearmanr(g.pred,g.act)[0])
r=np.array(r);print('n',len(r),'median rho',np.median(r),'min',r.min(),'q10',np.quantile(r,.1))
# Compression-matched regret: for each model-ds, among pipelines w/ compression >= c0, choose best by pred vs by actual
print('=== regret of picking best-predicted pipeline at compression >= c0 (among all combined pipelines, all_singles+ (m>p)xcodec-as-fallback)')
pp=A[A.pred_type.isin(['all_singles'])]
rows=[]
for c0 in [10,20,40]:
  reg=[];hit=0;tot=0;viol=[]
  for (m,d),g in pp.groupby(['model','ds']):
    g=g[g.comp>=c0]
    if len(g)<3: continue
    ch=g.loc[g.pred.idxmax()]; best=g.act.max()
    reg.append(best-ch.act); tot+=1; hit+=ch.act>=best-0.005; viol.append(ch.pred-ch.act)
  reg=np.array(reg);print(c0,'units',tot,'mean regret %.4f max %.4f'%(reg.mean(),reg.max()),'chosen within .005 of best: %.2f'%(hit/tot),'mean over-promise %.4f max %.4f'%(np.mean(viol),np.max(viol)))
