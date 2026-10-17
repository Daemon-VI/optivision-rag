import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
from scipy.stats import wilcoxon
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
rng=np.random.default_rng(2)
A=P[P.pred_type=='all_singles'].copy(); A['err']=A.act-A.pred
def auc(score,y):
    from scipy.stats import rankdata
    r=rankdata(score); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
print('=== AUC for act>=T by predictor (all_singles, n=%d)'%len(A))
A['min']=A.parts.map(min);A['first']=A.parts.map(lambda p:p[0]);A['add']=A.parts.map(lambda p:1-sum(1-x for x in p))
A['lastcodec']=A.parts.map(lambda p:p[-1])
for T in [0.95,0.98,0.99]:
    y=(A.act>=T).values
    print(T,{nm:round(auc(A[nm].values,y),3) for nm in ['pred','add','min','first','lastcodec']})
# noise ceiling: share of rows whose act CI straddles T
for T in [0.95,0.98,0.99]:
    st=((A.alo<T)&(A.ahi>T)); y=(A.act>=T)
    pred_c=(A.pred>=T)
    mis=(pred_c!=y)
    print(T,'CI straddles T: %.3f ; misclassified %.3f ; misclassified & straddling %.3f of misclassified'%(st.mean(),mis.mean(),(mis&st).sum()/max(1,mis.sum())))
# centring paired test
print('=== centred vs uncentred codec after merge: paired err difference (int4(mean) - int4) per merge/model/ds')
a=A[A.kinds=='merge>codec'].copy()
a['merge']=a.stages.map(lambda s:s[0]); a['codec']=a.stages.map(lambda s:s[1])
pv=a.pivot_table(index=['model','ds','merge'],columns='codec',values='err')
pv=pv.reset_index(); pv['grp']=np.where(pv.ds=='scifact','text','image')
for c1,c2 in [('int4(mean)','int4'),('int8(per_vector)','int4')]:
    for gr in ['image','text']:
        g=pv[pv.grp==gr].dropna(subset=[c1,c2]); d=(g[c1]-g[c2])
        mdl=g.assign(d=d).groupby('model').d.mean()
        bs=[rng.choice(mdl.values,len(mdl)).mean() for _ in range(3000)]
        print(c1,'-',c2,gr,'n',len(g),'models',len(mdl),'mean diff %.4f CI [%.4f,%.4f]'%(d.mean(),*np.percentile(bs,[2.5,97.5])),'wilcoxon p=%.2g'%wilcoxon(mdl.values).pvalue if len(mdl)>5 else '')
# matched-loss comparison binary/int4/lloyd2 image vs text
print('=== binary/int4/lloyd2 after merge: err vs single codec loss, by group, matched bins')
b=A[(A.kinds=='merge>codec')&A.stages.map(lambda s:s[1] in('binary','int4','lloyd2(7)'))].copy()
b['cl']=1-b.parts.map(lambda p:p[1]); b['grp']=np.where(b.ds=='scifact','text','image')
b['bin']=pd.cut(b.cl,[-1,0.01,0.03,0.08,0.2,1])
print(b.groupby(['bin','grp']).apply(lambda x:pd.Series(dict(n=len(x),models=x.model.nunique(),bias=x.err.mean(),rel=(x.err/np.maximum(x.cl,.003)).median()))).round(4))
