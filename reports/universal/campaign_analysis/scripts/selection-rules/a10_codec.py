import sys; sys.path.insert(0,'.')
from load import *
from scipy import stats
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',500)
O=pd.read_pickle(OUT+'/O.pkl')
L=O[(O.reference=='labels')&O.sel_ok].copy(); L['fail']=(L.hold<L.target).astype(float)
def codec(s):
    for c in ['binary','int4','int8','float16']:
        if c in s: return c
L['codec']=L.sel.map(codec); L['merge']=np.where(L.sel.str.contains('hierarchical'),'ward',np.where(L.sel.str.contains('adaptive'),'adaptive','none'))
print(L.groupby(['rule','codec']).fail.agg(['size','mean']).round(3).unstack(0).to_string())
print(L.groupby(['rule','merge']).fail.agg(['size','mean']).round(3).unstack(0).to_string())
# fail vs cal overshoot (cal - target) among default
for r in ['calibrate() default','lower 0.999','lower 0.975']:
    x=L[L.rule==r].copy(); x['ov']=x.cal-x.target
    x['b']=pd.qcut(x.ov,5,duplicates='drop'); print(r); print(x.groupby('b',observed=True).fail.agg(['size','mean']).round(3).to_string())
    print('spearman fail~overshoot',stats.spearmanr(x.fail,x.ov))
# share of selections by codec per rule
print(pd.crosstab(L.rule,L.codec,normalize='index').round(3))
# per-target selected-compression distribution default: fraction at 3.9 (int8) 
for t in [0.99,0.97,0.95]:
    x=L[(L.rule=='calibrate() default')&(L.target==t)]; print(t,'share int8/float16 picks',x.codec.isin(['int8','float16']).mean().round(3),'median comp',x.comp.median())
